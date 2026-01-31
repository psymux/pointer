#!/usr/bin/env python3
import argparse
import math
import sys
from urllib.parse import quote_plus

import collections
import pathlib

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs


EARTH_RADIUS_KM = 6371.0
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


def _smooth_noise(shape, scales=(6, 12, 24), seed=2):
    h, w = shape
    rng = np.random.default_rng(seed)
    noise = np.zeros((h, w), dtype=float)
    for scale in scales:
        scale = max(1, int(scale))
        gh = max(1, math.ceil(h / scale))
        gw = max(1, math.ceil(w / scale))
        grid = rng.random((gh, gw))
        up = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)
        noise += up[:h, :w]
    noise /= len(scales)

    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()
    noise = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, noise)
    noise = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, noise)
    return noise


def _earth_colors(lon, lat):
    shape = lon.shape
    noise = _smooth_noise(shape, seed=3)
    desert = _smooth_noise(shape, seed=7)

    ocean = np.array([0.05, 0.14, 0.32])
    shallow = np.array([0.08, 0.22, 0.45])
    land = np.array([0.08, 0.35, 0.12])
    desert_color = np.array([0.55, 0.45, 0.25])
    ice = np.array([0.9, 0.93, 0.95])

    colors = np.zeros((*shape, 3), dtype=float)
    colors[:] = ocean

    land_mask = noise > 0.53
    shallow_mask = (noise > 0.48) & ~land_mask
    colors[shallow_mask] = shallow
    colors[land_mask] = land

    desert_mask = land_mask & (desert > 0.58) & (np.abs(lat) < math.radians(35))
    colors[desert_mask] = desert_color

    ice_mask = np.abs(lat) > math.radians(70)
    colors[ice_mask] = ice

    clouds = _smooth_noise(shape, seed=11)
    cloud_mask = clouds > 0.76
    colors[cloud_mask] = colors[cloud_mask] * 0.7 + np.array([1.0, 1.0, 1.0]) * 0.3

    return colors


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


def _texture_colors(img, lon, lat):
    h, w = img.shape[:2]
    u = (lon / (2 * math.pi)) % 1.0
    v = 0.5 - (lat / math.pi)
    i = np.clip((v * (h - 1)).astype(int), 0, h - 1)
    j = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    return img[i, j]


def build_earth(ax, texture_img=None):
    u = np.linspace(0, 2 * math.pi, 160)
    v = np.linspace(0, math.pi, 80)
    lon, colat = np.meshgrid(u, v, indexing="ij")
    lat = (math.pi / 2) - colat

    x = EARTH_RADIUS_KM * np.cos(lon) * np.sin(colat)
    y = EARTH_RADIUS_KM * np.sin(lon) * np.sin(colat)
    z = EARTH_RADIUS_KM * np.cos(colat)

    if texture_img is not None:
        colors = _texture_colors(texture_img, lon, lat)
    else:
        colors = _earth_colors(lon, lat)

    light_dir = np.array([1.0, 0.2, 0.1])
    light_dir /= np.linalg.norm(light_dir)
    normals = np.stack((x, y, z), axis=-1) / EARTH_RADIUS_KM
    shade = np.clip(np.tensordot(normals, light_dir, axes=([2], [0])), 0, 1)
    shade = 0.25 + 0.75 * shade
    colors = colors * shade[..., None]

    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    # Atmospheric glow
    glow_r = EARTH_RADIUS_KM * 1.02
    ax.plot_surface(
        x * (glow_r / EARTH_RADIUS_KM),
        y * (glow_r / EARTH_RADIUS_KM),
        z * (glow_r / EARTH_RADIUS_KM),
        rstride=2,
        cstride=2,
        color="#6ec6ff",
        alpha=0.08,
        linewidth=0,
        shade=False,
    )


def set_axes_equal(ax, extent_km):
    ax.set_xlim(-extent_km, extent_km)
    ax.set_ylim(-extent_km, extent_km)
    ax.set_zlim(-extent_km, extent_km)
    ax.set_box_aspect([1, 1, 1])


def main():
    parser = argparse.ArgumentParser(description="Visualize a spacecraft from TLE.")
    parser.add_argument("--name", default="ISS (ZARYA)", help="Spacecraft name for TLE lookup.")
    parser.add_argument("--catnr", help="NORAD catalog number.")
    parser.add_argument("--update-seconds", type=float, default=1.0, help="Visualization update interval.")
    parser.add_argument("--extent-km", type=float, default=20000, help="Plot extent from Earth center.")
    parser.add_argument("--trail-minutes", type=float, default=45.0, help="Minutes of trail history to render.")
    parser.add_argument("--future-minutes", type=float, default=90.0, help="Minutes of future orbit to render.")
    parser.add_argument("--future-points", type=int, default=120, help="Number of samples for future orbit.")
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

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")

    texture_img = None
    try:
        texture_path = ensure_texture(args.earth_texture)
        texture_img = load_texture(texture_path)
    except Exception as exc:
        print(f"Warning: failed to load Earth texture: {exc}", file=sys.stderr)
        texture_img = None

    build_earth(ax, texture_img=texture_img)
    set_axes_equal(ax, args.extent_km)
    ax.set_axis_off()

    sat_scatter = ax.scatter([], [], [], color="#ff6b6b", s=25)
    sat_line, = ax.plot([0, 0], [0, 0], [0, 0], color="#ff6b6b", linewidth=1)
    trail_line, = ax.plot([], [], [], color="#ffa36c", linewidth=1.2, alpha=0.75)
    future_line, = ax.plot([], [], [], color="#6ee7ff", linewidth=1.0, alpha=0.7)

    trail_len = max(5, int(args.trail_minutes * 60 / args.update_seconds))
    trail = collections.deque(maxlen=trail_len)
    frame_counter = 0

    def update(_frame):
        nonlocal frame_counter
        t = ts.now()
        geocentric = sat.at(t)
        x, y, z = geocentric.frame_xyz(itrs).km

        sat_scatter._offsets3d = ([x], [y], [z])
        sat_line.set_data([0, x], [0, y])
        sat_line.set_3d_properties([0, z])

        trail.append((x, y, z))
        if len(trail) > 2:
            tx, ty, tz = zip(*trail)
            trail_line.set_data(tx, ty)
            trail_line.set_3d_properties(tz)

        if frame_counter % 10 == 0:
            future_seconds = np.linspace(0, args.future_minutes * 60, args.future_points)
            future_times = t + future_seconds / 86400.0
            future_geo = sat.at(future_times)
            fx, fy, fz = future_geo.frame_xyz(itrs).km
            future_line.set_data(fx, fy)
            future_line.set_3d_properties(fz)

        frame_counter += 1

        subpoint = wgs84.subpoint(geocentric)
        title = (
            f"{sat.name}  "
            f"lat {subpoint.latitude.degrees:+.2f}°  "
            f"lon {subpoint.longitude.degrees:+.2f}°  "
            f"alt {subpoint.elevation.km:.1f} km"
        )
        ax.set_title(title, color="white")
        ax.view_init(elev=20, azim=(frame_counter * 0.25) % 360)
        return sat_scatter, sat_line, trail_line, future_line

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
