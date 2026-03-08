#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import sys
import time
from urllib.parse import quote_plus

from dynamixel_sdk import PacketHandler, PortHandler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
from skyfield.api import EarthSatellite

from pointer_targets import (
    AmbiguousTargetError,
    TargetResolutionError,
    TargetResolver,
    TargetSpec,
)
from pointer_web import serve_web_app


TEXTURE_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a003100/a003191/frames/2048x1024/background-bluemarble.png"
DEFAULT_POINTER_CONFIG_PATH = "pointer_config.json"

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

OPERATING_MODE_POSITION = 3
OPERATING_MODE_EXTENDED_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
PROFILE_VELOCITY_RPM_PER_UNIT = 0.229


def fetch_tle(name=None, catnr=None, timeout=10):
    if name and catnr:
        raise ValueError("Provide either name or catnr, not both.")
    if not name and not catnr:
        raise ValueError("Provide a spacecraft name or a NORAD catalog number.")

    if catnr:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={int(catnr)}&FORMAT=TLE"
    else:
        encoded = quote_plus(name.strip())
        url = f"https://celestrak.org/NORAD/elements/gp.php?NAME={encoded}&FORMAT=TLE"

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"No TLE found for {name or catnr}.")

    return lines[0], lines[1], lines[2]


def ensure_texture(texture_path, url=TEXTURE_URL, timeout=20):
    texture_path = pathlib.Path(texture_path)
    if texture_path.exists():
        return texture_path
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64)"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    texture_path.write_bytes(resp.content)
    return texture_path


def load_texture(texture_path):
    img = mpimg.imread(texture_path)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def wrap_lon(lon):
    return ((lon + 180) % 360) - 180


def wrap_degrees_180(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0


def normalize_degrees_360(angle_deg):
    return angle_deg % 360.0


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def degrees_per_second_to_profile_velocity_units(deg_per_s):
    rpm = float(deg_per_s) / 6.0
    units = int(round(rpm / PROFILE_VELOCITY_RPM_PER_UNIT))
    return max(1, min(32767, units))


def circular_distance_deg(a, b):
    return abs(wrap_degrees_180(a - b))


def clamp_azimuth(az_deg, az_min_deg, az_max_deg):
    az = normalize_degrees_360(az_deg)
    az_min = normalize_degrees_360(az_min_deg)
    az_max = normalize_degrees_360(az_max_deg)

    if az_min <= az_max:
        clamped = clamp(az, az_min, az_max)
        return clamped, abs(clamped - az) > 1e-9

    if az >= az_min or az <= az_max:
        return az, False

    dist_to_min = circular_distance_deg(az, az_min)
    dist_to_max = circular_distance_deg(az, az_max)
    clamped = az_min if dist_to_min <= dist_to_max else az_max
    return clamped, True


def trail_with_gaps(lons, lats, jump_threshold=180.0):
    if len(lons) < 2:
        return lons, lats
    out_lons = [lons[0]]
    out_lats = [lats[0]]
    for prev_lon, lon, lat in zip(lons[:-1], lons[1:], lats[1:]):
        if abs(lon - prev_lon) > jump_threshold:
            out_lons.append(np.nan)
            out_lats.append(np.nan)
        out_lons.append(lon)
        out_lats.append(lat)
    return out_lons, out_lats


def signed_int32(value):
    if value & 0x80000000:
        return value - 0x100000000
    return value


def deep_update(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def default_pointer_config():
    return {
        "version": 1,
        "observer": {
            "lat": -36.8485,
            "lon": 174.7633,
            "elevation_m": 0.0,
        },
        "servo": {
            "port": "/dev/ttyACM0",
            "baud": 115200,
            "base_id": 2,
            "alt_id": 3,
            "ticks_per_rev": 4096.0,
            "base_slew_deg_per_s": 20.0,
            "alt_slew_deg_per_s": 20.0,
            "base_dir": 1,
            "alt_dir": 1,
            "az_min_deg": 0.0,
            "az_max_deg": 90.0,
            "alt_min_deg": -90.0,
            "alt_max_deg": 90.0,
            "az_offset_deg": 0.0,
            "alt_offset_deg": 0.0,
            "base_reference_ticks": None,
            "alt_reference_ticks": None,
            "base_reference_az_deg": 0.0,
            "alt_reference_alt_deg": 0.0,
        },
    }


def load_pointer_config(path):
    config = default_pointer_config()
    if not path.exists():
        return config

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config at {path} is not a JSON object.")

    deep_update(config, payload)
    return config


def save_pointer_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def apply_cli_overrides(config, args):
    observer = config["observer"]
    servo = config["servo"]

    if args.observer_lat is not None:
        observer["lat"] = float(args.observer_lat)
    if args.observer_lon is not None:
        observer["lon"] = float(args.observer_lon)
    if args.observer_elevation_m is not None:
        observer["elevation_m"] = float(args.observer_elevation_m)

    if args.servo_port is not None:
        servo["port"] = args.servo_port
    if args.servo_baud is not None:
        servo["baud"] = int(args.servo_baud)
    if args.base_servo_id is not None:
        servo["base_id"] = int(args.base_servo_id)
    if args.alt_servo_id is not None:
        servo["alt_id"] = int(args.alt_servo_id)
    if args.ticks_per_rev is not None:
        servo["ticks_per_rev"] = float(args.ticks_per_rev)
    if args.base_dir is not None:
        servo["base_dir"] = int(args.base_dir)
    if args.alt_dir is not None:
        servo["alt_dir"] = int(args.alt_dir)
    if args.base_slew_deg_per_s is not None:
        servo["base_slew_deg_per_s"] = float(args.base_slew_deg_per_s)
    if args.alt_slew_deg_per_s is not None:
        servo["alt_slew_deg_per_s"] = float(args.alt_slew_deg_per_s)
    if args.az_min_deg is not None:
        servo["az_min_deg"] = float(args.az_min_deg)
    if args.az_max_deg is not None:
        servo["az_max_deg"] = float(args.az_max_deg)
    if args.alt_min_deg is not None:
        servo["alt_min_deg"] = float(args.alt_min_deg)
    if args.alt_max_deg is not None:
        servo["alt_max_deg"] = float(args.alt_max_deg)
    if args.az_offset_deg is not None:
        servo["az_offset_deg"] = float(args.az_offset_deg)
    if args.alt_offset_deg is not None:
        servo["alt_offset_deg"] = float(args.alt_offset_deg)


class DxlPointerController:
    def __init__(
        self,
        *,
        port,
        baud,
        base_id,
        alt_id,
        ticks_per_rev,
        base_slew_deg_per_s,
        alt_slew_deg_per_s,
        base_dir,
        alt_dir,
        az_min_deg,
        az_max_deg,
        alt_min_deg,
        alt_max_deg,
        az_offset_deg,
        alt_offset_deg,
        base_reference_ticks,
        alt_reference_ticks,
        base_reference_az_deg,
        alt_reference_alt_deg,
    ):
        self.port = port
        self.baud = int(baud)
        self.base_id = int(base_id)
        self.alt_id = int(alt_id)
        self.ticks_per_deg = float(ticks_per_rev) / 360.0
        self.base_profile_velocity = degrees_per_second_to_profile_velocity_units(base_slew_deg_per_s)
        self.alt_profile_velocity = degrees_per_second_to_profile_velocity_units(alt_slew_deg_per_s)
        self.base_dir = int(base_dir)
        self.alt_dir = int(alt_dir)
        self.az_min_deg = float(az_min_deg)
        self.az_max_deg = float(az_max_deg)
        self.alt_min_deg = float(alt_min_deg)
        self.alt_max_deg = float(alt_max_deg)
        self.az_offset_deg = float(az_offset_deg)
        self.alt_offset_deg = float(alt_offset_deg)
        self.base_reference_ticks = (
            None if base_reference_ticks is None else int(base_reference_ticks)
        )
        self.alt_reference_ticks = None if alt_reference_ticks is None else int(alt_reference_ticks)
        self.base_reference_az_deg = float(base_reference_az_deg)
        self.alt_reference_alt_deg = float(alt_reference_alt_deg)

        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(2.0)

    def open(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            raise RuntimeError(f"Failed to set baud {self.baud}")

        self._set_mode(self.base_id, OPERATING_MODE_POSITION)
        self._set_mode(self.alt_id, OPERATING_MODE_EXTENDED_POSITION)
        self._write4(self.base_id, ADDR_PROFILE_VELOCITY, self.base_profile_velocity)
        self._write4(self.alt_id, ADDR_PROFILE_VELOCITY, self.alt_profile_velocity)

    def close(self):
        self.port_handler.closePort()

    def _write1(self, dxl_id, address, value):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, address, value
        )
        if dxl_comm_result != 0 or dxl_error != 0:
            raise RuntimeError(
                f"write1 failed id={dxl_id} addr={address} comm={dxl_comm_result} err={dxl_error}"
            )

    def _write4(self, dxl_id, address, value):
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, address, value
        )
        if dxl_comm_result != 0 or dxl_error != 0:
            raise RuntimeError(
                f"write4 failed id={dxl_id} addr={address} comm={dxl_comm_result} err={dxl_error}"
            )

    def _read4(self, dxl_id, address, *, signed):
        value, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, address
        )
        if dxl_comm_result != 0 or dxl_error != 0:
            raise RuntimeError(
                f"read4 failed id={dxl_id} addr={address} comm={dxl_comm_result} err={dxl_error}"
            )
        return signed_int32(value) if signed else value

    def _set_mode(self, dxl_id, operating_mode):
        self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self._write1(dxl_id, ADDR_OPERATING_MODE, operating_mode)
        self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    def set_torque_enabled(self, enabled):
        torque_value = TORQUE_ENABLE if enabled else TORQUE_DISABLE
        self._write1(self.base_id, ADDR_TORQUE_ENABLE, torque_value)
        self._write1(self.alt_id, ADDR_TORQUE_ENABLE, torque_value)

    def read_present_positions(self):
        base_pos = self._read4(self.base_id, ADDR_PRESENT_POSITION, signed=False)
        alt_pos = self._read4(self.alt_id, ADDR_PRESENT_POSITION, signed=True)
        return base_pos, alt_pos

    def set_reference(self, *, base_ticks, alt_ticks, base_az_deg, alt_deg):
        self.base_reference_ticks = int(base_ticks)
        self.alt_reference_ticks = int(alt_ticks)
        self.base_reference_az_deg = float(base_az_deg)
        self.alt_reference_alt_deg = float(alt_deg)

    def point(self, az_deg, alt_deg):
        if self.base_reference_ticks is None or self.alt_reference_ticks is None:
            raise RuntimeError("Missing reference ticks; run --set-reference first.")

        az_raw = normalize_degrees_360(float(az_deg) + self.az_offset_deg)
        az_cmd_deg, az_clamped = clamp_azimuth(az_raw, self.az_min_deg, self.az_max_deg)
        preferred_az_delta_deg = wrap_degrees_180(az_cmd_deg - self.base_reference_az_deg)
        preferred_base_goal = int(
            round(
                self.base_reference_ticks
                + self.base_dir * preferred_az_delta_deg * self.ticks_per_deg
            )
        )

        base_goal = None
        for wrap_turns in (0, -1, 1, -2, 2):
            az_delta_deg = preferred_az_delta_deg + (360.0 * wrap_turns)
            candidate_base_goal = int(
                round(
                    self.base_reference_ticks
                    + self.base_dir * az_delta_deg * self.ticks_per_deg
                )
            )
            if 0 <= candidate_base_goal <= 4095:
                base_goal = candidate_base_goal
                break

        if base_goal is None:
            base_goal = int(clamp(preferred_base_goal, 0, 4095))
            az_clamped = True

        alt_raw = float(alt_deg) + self.alt_offset_deg
        alt_cmd_deg = clamp(alt_raw, self.alt_min_deg, self.alt_max_deg)
        alt_clamped = abs(alt_cmd_deg - alt_raw) > 1e-9
        alt_delta_deg = alt_cmd_deg - self.alt_reference_alt_deg
        alt_goal = int(round(self.alt_reference_ticks + self.alt_dir * alt_delta_deg * self.ticks_per_deg))

        self._write4(self.base_id, ADDR_GOAL_POSITION, base_goal)
        self._write4(self.alt_id, ADDR_GOAL_POSITION, alt_goal & 0xFFFFFFFF)

        return {
            "az_deg": float(az_deg),
            "alt_deg": float(alt_deg),
            "az_cmd_deg": az_cmd_deg,
            "alt_cmd_deg": alt_cmd_deg,
            "az_clamped": az_clamped,
            "alt_clamped": alt_clamped,
            "base_goal": base_goal,
            "alt_goal": alt_goal,
        }


def build_pointer_controller(config):
    servo = config["servo"]
    return DxlPointerController(
        port=servo["port"],
        baud=servo["baud"],
        base_id=servo["base_id"],
        alt_id=servo["alt_id"],
        ticks_per_rev=servo["ticks_per_rev"],
        base_slew_deg_per_s=servo.get("base_slew_deg_per_s", 20.0),
        alt_slew_deg_per_s=servo.get("alt_slew_deg_per_s", 20.0),
        base_dir=servo["base_dir"],
        alt_dir=servo["alt_dir"],
        az_min_deg=servo["az_min_deg"],
        az_max_deg=servo["az_max_deg"],
        alt_min_deg=servo["alt_min_deg"],
        alt_max_deg=servo["alt_max_deg"],
        az_offset_deg=servo["az_offset_deg"],
        alt_offset_deg=servo["alt_offset_deg"],
        base_reference_ticks=servo.get("base_reference_ticks"),
        alt_reference_ticks=servo.get("alt_reference_ticks"),
        base_reference_az_deg=servo.get("base_reference_az_deg", 0.0),
        alt_reference_alt_deg=servo.get("alt_reference_alt_deg", 0.0),
    )


def run_startup_home(pointer, hold_seconds):
    cmd = pointer.point(0.0, 0.0)
    print(
        f"Startup home: az=0.00° alt=0.00° "
        f"(base_goal={cmd['base_goal']} alt_goal={cmd['alt_goal']})"
    )
    if hold_seconds > 0:
        time.sleep(hold_seconds)


def build_target_spec_from_args(args):
    if args.target:
        return TargetSpec(
            kind=args.target_kind,
            query=args.target,
            source=args.target_source,
            identifier=args.target_id,
        )
    if args.catnr:
        return TargetSpec(kind="satellite", query=str(args.catnr), identifier=str(args.catnr))
    if args.name:
        return TargetSpec(kind="satellite", query=args.name)
    return TargetSpec(kind="satellite", query="ISS (ZARYA)")


def print_target_matches(matches):
    if not matches:
        print("No matches found.")
        return
    for index, match in enumerate(matches, start=1):
        detail = f" id={match.identifier}" if match.identifier else ""
        source = f" source={match.source}" if match.source else ""
        description = f" - {match.description}" if match.description else ""
        print(f"{index:02d}. [{match.kind}] {match.display_name}{detail}{source}{description}")


def create_earth_plot(ax, observer_cfg, texture_path):
    ax.set_facecolor("#000000")
    ax.figure.patch.set_facecolor("#000000")

    texture_img = None
    try:
        texture = ensure_texture(texture_path)
        texture_img = load_texture(texture)
    except Exception as exc:
        print(f"Warning: failed to load Earth texture: {exc}", file=sys.stderr)

    if texture_img is not None:
        ax.imshow(
            np.flipud(texture_img),
            extent=(-180, 180, -90, 90),
            origin="lower",
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    trail_line, = ax.plot([], [], color="#ffa36c", linewidth=1.4, alpha=0.85, zorder=3)
    target_scatter = ax.scatter([], [], color="#ff4d4d", s=30, zorder=4)
    ax.scatter(
        [wrap_lon(observer_cfg["lon"])],
        [observer_cfg["lat"]],
        color="#4da6ff",
        marker="^",
        s=55,
        zorder=5,
    )
    return {"trail_line": trail_line, "target_scatter": target_scatter}


def create_sky_plot(ax):
    ax.set_facecolor("#05080f")
    ax.figure.patch.set_facecolor("#05080f")
    ax.set_xlim(0, 360)
    ax.set_ylim(-20, 90)
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Altitude (deg)")
    ax.axhline(0, color="#5d7384", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.grid(color="#213642", alpha=0.45, linewidth=0.8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    trail_line, = ax.plot([], [], color="#70e000", linewidth=1.4, alpha=0.85)
    target_scatter = ax.scatter([], [], color="#ffd166", s=45, zorder=4)
    return {"trail_line": trail_line, "target_scatter": target_scatter}


def update_plot_artists(artists, state, trail):
    if state.plot_mode == "earth":
        lon = wrap_lon(state.metadata["subpoint_lon_deg"])
        lat = state.metadata["subpoint_lat_deg"]
        artists["target_scatter"].set_offsets([[lon, lat]])
        trail.append((lon, lat))
        if len(trail) > 1:
            lons, lats = zip(*trail)
            lons, lats = trail_with_gaps(list(lons), list(lats))
            artists["trail_line"].set_data(lons, lats)
        return artists["target_scatter"], artists["trail_line"]

    az = float(state.az_deg)
    alt = float(state.alt_deg)
    artists["target_scatter"].set_offsets([[az, alt]])
    trail.append((az, alt))
    if len(trail) > 1:
        azs, alts = zip(*trail)
        azs, alts = trail_with_gaps(list(azs), list(alts))
        artists["trail_line"].set_data(azs, alts)
    return artists["target_scatter"], artists["trail_line"]


def format_distance(state):
    if state.distance is None or not state.distance_unit:
        return "n/a"
    return f"{state.distance:.3f} {state.distance_unit}"


def main():
    parser = argparse.ArgumentParser(description="Visualize and point at sky targets using DYNAMIXEL servos.")
    parser.add_argument("--config", default=DEFAULT_POINTER_CONFIG_PATH, help="Path to persistent pointer config JSON.")
    parser.add_argument("--show-config", action="store_true", help="Print resolved config and exit.")

    parser.add_argument("--set-reference", action="store_true", help="Save current servo pose as known az/alt.")
    parser.add_argument(
        "--calibrate-north-horizon",
        action="store_true",
        help="Disable torque, let user manually align north/horizon, then save az=0 alt=0 reference.",
    )
    parser.add_argument("--reference-az-deg", type=float, help="Known azimuth at current pose for --set-reference.")
    parser.add_argument("--reference-alt-deg", type=float, help="Known altitude at current pose for --set-reference.")
    parser.add_argument("--point-az-deg", type=float, help="One-shot absolute azimuth command (degrees).")
    parser.add_argument("--point-alt-deg", type=float, help="One-shot absolute altitude command (degrees).")
    parser.add_argument(
        "--startup-hold-seconds",
        type=float,
        default=3.0,
        help="Hold time after startup homing to north/horizon before target pointing.",
    )
    parser.add_argument("--target", help="Target name, catalog entry, or object identifier.")
    parser.add_argument(
        "--target-kind",
        default="auto",
        choices=("auto", "satellite", "solar-system", "spacecraft", "star", "constellation", "dso"),
        help="Target class to resolve.",
    )
    parser.add_argument("--target-id", help="Optional explicit provider identifier, e.g. NORAD or Horizons SPK-ID.")
    parser.add_argument("--target-source", help="Optional provider source hint.")
    parser.add_argument("--search", dest="search_query", help="Search for matching targets and exit.")
    parser.add_argument("--list-matches", dest="search_query", help="Alias for --search.")
    parser.add_argument("--name", default=None, help="Deprecated satellite name alias.")
    parser.add_argument("--catnr", help="Deprecated NORAD catalog number alias.")
    parser.add_argument("--update-seconds", type=float, default=1.0, help="Visualization update interval.")
    parser.add_argument("--trail-minutes", type=float, default=60.0, help="Minutes of trail history to render.")
    parser.add_argument(
        "--earth-texture",
        default="assets/earth_2048.jpg",
        help="Path to an equirectangular Earth texture (will download if missing).",
    )
    parser.add_argument("--cache-dir", default=None, help="Cache directory for catalogs and ephemerides.")
    parser.add_argument("--serve", action="store_true", help="Start the local web control interface.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for --serve.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve.")
    parser.add_argument("--preset-store", default=None, help="JSON file for saved web presets.")

    parser.add_argument("--observer-lat", type=float, default=None, help="Override observer latitude.")
    parser.add_argument("--observer-lon", type=float, default=None, help="Override observer longitude.")
    parser.add_argument("--observer-elevation-m", type=float, default=None, help="Override observer elevation (m).")

    parser.add_argument("--servo-port", default=None, help="Override serial device path.")
    parser.add_argument("--servo-baud", type=int, default=None, help="Override servo bus baud rate.")
    parser.add_argument("--base-servo-id", type=int, default=None, help="Override base (azimuth) servo ID.")
    parser.add_argument("--alt-servo-id", type=int, default=None, help="Override altitude (pitch) servo ID.")
    parser.add_argument("--ticks-per-rev", type=float, default=None, help="Override encoder ticks per revolution.")
    parser.add_argument("--base-dir", type=int, choices=(-1, 1), default=None, help="Azimuth direction sign.")
    parser.add_argument("--alt-dir", type=int, choices=(-1, 1), default=None, help="Altitude direction sign.")
    parser.add_argument("--base-slew-deg-per-s", type=float, default=None, help="Azimuth slew rate limit in degrees/sec.")
    parser.add_argument("--alt-slew-deg-per-s", type=float, default=None, help="Altitude slew rate limit in degrees/sec.")
    parser.add_argument("--az-min-deg", type=float, default=None, help="Azimuth minimum command limit.")
    parser.add_argument("--az-max-deg", type=float, default=None, help="Azimuth maximum command limit.")
    parser.add_argument("--alt-min-deg", type=float, default=None, help="Altitude minimum command limit.")
    parser.add_argument("--alt-max-deg", type=float, default=None, help="Altitude maximum command limit.")
    parser.add_argument("--az-offset-deg", type=float, default=None, help="Azimuth offset calibration.")
    parser.add_argument("--alt-offset-deg", type=float, default=None, help="Altitude offset calibration.")
    parser.add_argument("--disable-servo", action="store_true", help="Disable servo output.")
    args = parser.parse_args()

    resolver = TargetResolver(cache_dir=pathlib.Path(args.cache_dir).expanduser() if args.cache_dir else None)
    config_path = pathlib.Path(args.config)
    try:
        config = load_pointer_config(config_path)
    except Exception as exc:
        print(f"Error loading config {config_path}: {exc}", file=sys.stderr)
        return 1

    apply_cli_overrides(config, args)
    observer_cfg = config["observer"]
    servo_cfg = config["servo"]

    if args.show_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0

    if args.search_query:
        try:
            matches = resolver.search(args.search_query, kind=args.target_kind)
            print_target_matches(matches)
            return 0
        except Exception as exc:
            print(f"Error searching targets: {exc}", file=sys.stderr)
            return 1

    if args.serve:
        preset_store = (
            pathlib.Path(args.preset_store).expanduser()
            if args.preset_store
            else resolver.cache_dir / "presets.json"
        )
        try:
            serve_web_app(
                resolver=resolver,
                observer_cfg=observer_cfg,
                pointer_factory=build_pointer_controller,
                pointer_config=config,
                disable_servo=args.disable_servo,
                host=args.host,
                port=args.port,
                update_seconds=args.update_seconds,
                preset_store=preset_store,
            )
            return 0
        except Exception as exc:
            print(f"Error starting web interface: {exc}", file=sys.stderr)
            return 1

    pointer = None

    if args.calibrate_north_horizon:
        if args.disable_servo:
            print("Cannot use --calibrate-north-horizon with --disable-servo.", file=sys.stderr)
            return 2
        try:
            pointer = build_pointer_controller(config)
            pointer.open()
            pointer.set_torque_enabled(False)
            print("Calibration mode: torque disabled on both servos.")
            print("Manually point the pointer due north at 0 degrees altitude, then press Enter.")
            try:
                input()
            except EOFError:
                print("No input received; calibration aborted.", file=sys.stderr)
                return 2

            base_now, alt_now = pointer.read_present_positions()
            pointer.set_reference(
                base_ticks=base_now,
                alt_ticks=alt_now,
                base_az_deg=0.0,
                alt_deg=0.0,
            )
            servo_cfg["base_reference_ticks"] = pointer.base_reference_ticks
            servo_cfg["alt_reference_ticks"] = pointer.alt_reference_ticks
            servo_cfg["base_reference_az_deg"] = pointer.base_reference_az_deg
            servo_cfg["alt_reference_alt_deg"] = pointer.alt_reference_alt_deg
            save_pointer_config(config_path, config)
            pointer.set_torque_enabled(True)
            print(f"Saved north/horizon reference to {config_path}")
            print(
                f"Base ID {servo_cfg['base_id']} ticks={base_now} at az=0.00°, "
                f"Alt ID {servo_cfg['alt_id']} ticks={alt_now} at alt=0.00°"
            )
            return 0
        except Exception as exc:
            print(f"Error in calibration mode: {exc}", file=sys.stderr)
            return 1
        finally:
            if pointer is not None:
                pointer.close()

    if args.set_reference:
        if args.reference_az_deg is None or args.reference_alt_deg is None:
            print("--set-reference requires --reference-az-deg and --reference-alt-deg.", file=sys.stderr)
            return 2

        try:
            pointer = build_pointer_controller(config)
            pointer.open()
            base_now, alt_now = pointer.read_present_positions()
            pointer.set_reference(
                base_ticks=base_now,
                alt_ticks=alt_now,
                base_az_deg=args.reference_az_deg,
                alt_deg=args.reference_alt_deg,
            )
            servo_cfg["base_reference_ticks"] = pointer.base_reference_ticks
            servo_cfg["alt_reference_ticks"] = pointer.alt_reference_ticks
            servo_cfg["base_reference_az_deg"] = pointer.base_reference_az_deg
            servo_cfg["alt_reference_alt_deg"] = pointer.alt_reference_alt_deg
            save_pointer_config(config_path, config)
            print(f"Saved reference to {config_path}")
            print(
                f"Base ID {servo_cfg['base_id']} ticks={base_now} at az={args.reference_az_deg:.2f}°, "
                f"Alt ID {servo_cfg['alt_id']} ticks={alt_now} at alt={args.reference_alt_deg:.2f}°"
            )
            return 0
        except Exception as exc:
            print(f"Error setting reference: {exc}", file=sys.stderr)
            return 1
        finally:
            if pointer is not None:
                pointer.close()

    point_mode = (args.point_az_deg is not None) or (args.point_alt_deg is not None)
    if point_mode:
        if args.point_az_deg is None or args.point_alt_deg is None:
            print("--point-az-deg and --point-alt-deg must be provided together.", file=sys.stderr)
            return 2
        if args.disable_servo:
            print("Cannot use point mode with --disable-servo.", file=sys.stderr)
            return 2

        try:
            pointer = build_pointer_controller(config)
            pointer.open()
            run_startup_home(pointer, args.startup_hold_seconds)
            cmd = pointer.point(args.point_az_deg, args.point_alt_deg)
            print(
                f"Pointed to az={cmd['az_deg']:.2f}° alt={cmd['alt_deg']:.2f}° "
                f"(base_goal={cmd['base_goal']} alt_goal={cmd['alt_goal']})"
            )
            return 0
        except Exception as exc:
            print(f"Error in one-shot point mode: {exc}", file=sys.stderr)
            return 1
        finally:
            if pointer is not None:
                pointer.close()

    target_spec = build_target_spec_from_args(args)
    observer_context = resolver.build_observer_context(observer_cfg)
    try:
        target = resolver.resolve(target_spec, observer_cfg)
    except AmbiguousTargetError as exc:
        print(str(exc), file=sys.stderr)
        print_target_matches(exc.matches)
        return 2
    except Exception as exc:
        print(f"Error resolving target: {exc}", file=sys.stderr)
        return 1

    if not args.disable_servo:
        if servo_cfg.get("base_reference_ticks") is None or servo_cfg.get("alt_reference_ticks") is None:
            print(
                f"Missing reference ticks in {config_path}. "
                "Run --set-reference with a known az/alt first.",
                file=sys.stderr,
            )
            return 2
        try:
            pointer = build_pointer_controller(config)
            pointer.open()
            print(
                f"Servo mapping loaded from {config_path}: "
                f"base id={servo_cfg['base_id']} ref_ticks={servo_cfg['base_reference_ticks']} "
                f"at az={servo_cfg['base_reference_az_deg']:.2f}°, "
                f"alt id={servo_cfg['alt_id']} ref_ticks={servo_cfg['alt_reference_ticks']} "
                f"at alt={servo_cfg['alt_reference_alt_deg']:.2f}°"
            )
            run_startup_home(pointer, args.startup_hold_seconds)
        except Exception as exc:
            print(f"Error initializing servos: {exc}", file=sys.stderr)
            if pointer is not None:
                pointer.close()
            return 1

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    artists = create_earth_plot(ax, observer_cfg, args.earth_texture) if target.kind == "satellite" else create_sky_plot(ax)

    trail_len = max(5, int(args.trail_minutes * 60 / args.update_seconds))
    trail = collections.deque(maxlen=trail_len)

    def update(_frame):
        nonlocal target, artists
        t = resolver.ts.now()
        try:
            state = target.state_at(t, observer_context)
        except TargetResolutionError:
            target = resolver.resolve(target_spec, observer_cfg)
            state = target.state_at(t, observer_context)

        if state.plot_mode == "earth" and ax.get_xlabel() != "Longitude (deg)":
            ax.clear()
            artists = create_earth_plot(ax, observer_cfg, args.earth_texture)
            trail.clear()
        elif state.plot_mode == "sky" and ax.get_xlabel() != "Azimuth (deg)":
            ax.clear()
            artists = create_sky_plot(ax)
            trail.clear()

        update_plot_artists(artists, state, trail)

        servo_text = "servos disabled"
        if pointer is not None:
            try:
                cmd = pointer.point(az_deg=state.az_deg, alt_deg=state.alt_deg)
                clamp_bits = []
                if cmd["az_clamped"]:
                    clamp_bits.append("az")
                if cmd["alt_clamped"]:
                    clamp_bits.append("alt")
                clamp_suffix = f" clamped({','.join(clamp_bits)})" if clamp_bits else ""
                servo_text = (
                    f"servo az {cmd['az_cmd_deg']:+.1f}° -> {cmd['base_goal']}  "
                    f"alt {cmd['alt_cmd_deg']:+.1f}° -> {cmd['alt_goal']}{clamp_suffix}"
                )
            except Exception as exc:
                servo_text = f"servo error: {exc}"

        if state.plot_mode == "earth":
            title = (
                f"{state.display_name}  "
                f"lat {state.metadata['subpoint_lat_deg']:+.2f}°  "
                f"lon {wrap_lon(state.metadata['subpoint_lon_deg']):+.2f}°  "
                f"alt {state.metadata['subpoint_elevation_km']:.1f} km\n"
                f"Observer az {state.az_deg:+.1f}°  el {state.alt_deg:+.1f}°  {servo_text}"
            )
        else:
            title = (
                f"{state.display_name} [{state.kind}]  "
                f"RA {state.ra_deg/15.0:05.2f}h  "
                f"Dec {state.dec_deg:+.2f}°  "
                f"Dist {format_distance(state)}\n"
                f"Observer az {state.az_deg:+.1f}°  el {state.alt_deg:+.1f}°  {servo_text}"
            )
        ax.set_title(title, color="white")
        return artists["target_scatter"], artists["trail_line"]

    anim = animation.FuncAnimation(
        fig,
        update,
        interval=int(args.update_seconds * 1000),
        cache_frame_data=False,
        blit=False,
    )
    try:
        plt.show()
    finally:
        if pointer is not None:
            pointer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
