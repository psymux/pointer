#!/usr/bin/env python3
import argparse
import collections
import math
import pathlib
import sys
from urllib.parse import quote_plus

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
from skyfield.api import EarthSatellite, load, wgs84


TEXTURE_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a003100/a003191/frames/2048x1024/background-bluemarble.png"


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

    # Some queries return multiple TLEs; take the first set.
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


def main():
    parser = argparse.ArgumentParser(description="Visualize a spacecraft from TLE (2D map).")
    parser.add_argument("--name", default="ISS (ZARYA)", help="Spacecraft name for TLE lookup.")
    parser.add_argument("--catnr", help="NORAD catalog number.")
    parser.add_argument("--update-seconds", type=float, default=1.0, help="Visualization update interval.")
    parser.add_argument("--trail-minutes", type=float, default=60.0, help="Minutes of trail history to render.")
    parser.add_argument(
        "--earth-texture",
        default="assets/earth_2048.jpg",
        help="Path to an equirectangular Earth texture (will download if missing).",
    )
    args = parser.parse_args()

    try:
        tle0, tle1, tle2 = fetch_tle(name=args.name, catnr=args.catnr)
    except Exception as exc:
        print(f"Error fetching TLE: {exc}", file=sys.stderr)
        return 1

    ts = load.timescale()
    sat = EarthSatellite(tle1, tle2, tle0, ts)

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")

    texture_img = None
    try:
        texture_path = ensure_texture(args.earth_texture)
        texture_img = load_texture(texture_path)
    except Exception as exc:
        print(f"Warning: failed to load Earth texture: {exc}", file=sys.stderr)
        texture_img = None

    if texture_img is not None:
        texture_img = np.flipud(texture_img)
        ax.imshow(
            texture_img,
            extent=(-180, 180, -90, 90),
            origin="lower",
        )
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    sat_scatter = ax.scatter([], [], color="#ff4d4d", s=30, zorder=4)
    trail_line, = ax.plot([], [], color="#ffa36c", linewidth=1.4, alpha=0.85, zorder=3)

    trail_len = max(5, int(args.trail_minutes * 60 / args.update_seconds))
    trail = collections.deque(maxlen=trail_len)

    def update(_frame):
        t = ts.now()
        geocentric = sat.at(t)
        subpoint = wgs84.subpoint(geocentric)
        lat = subpoint.latitude.degrees
        lon = wrap_lon(subpoint.longitude.degrees)

        sat_scatter.set_offsets([[lon, lat]])
        trail.append((lon, lat))
        if len(trail) > 2:
            lons, lats = zip(*trail)
            lons, lats = trail_with_gaps(list(lons), list(lats))
            trail_line.set_data(lons, lats)

        title = (
            f"{sat.name}  "
            f"lat {lat:+.2f}°  "
            f"lon {lon:+.2f}°  "
            f"alt {subpoint.elevation.km:.1f} km"
        )
        ax.set_title(title, color="white")
        return sat_scatter, trail_line

    anim = animation.FuncAnimation(
        fig,
        update,
        interval=int(args.update_seconds * 1000),
        cache_frame_data=False,
        blit=False,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
