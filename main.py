#!/usr/bin/env python3
import argparse
import math
import sys
from urllib.parse import quote_plus

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import requests
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs


EARTH_RADIUS_KM = 6371.0


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


def build_earth(ax):
    u = np.linspace(0, 2 * math.pi, 60)
    v = np.linspace(0, math.pi, 30)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=2, cstride=2, color="#3b6ea5", alpha=0.55, linewidth=0)


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
    args = parser.parse_args()

    try:
        tle0, tle1, tle2 = fetch_tle(name=args.name, catnr=args.catnr)
    except Exception as exc:
        print(f"Error fetching TLE: {exc}", file=sys.stderr)
        return 1

    ts = load.timescale()
    sat = EarthSatellite(tle1, tle2, tle0, ts)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b1020")
    fig.patch.set_facecolor("#0b1020")
    build_earth(ax)
    set_axes_equal(ax, args.extent_km)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    sat_scatter = ax.scatter([], [], [], color="#ff6b6b", s=25)
    sat_line, = ax.plot([0, 0], [0, 0], [0, 0], color="#ff6b6b", linewidth=1)

    def update(_frame):
        t = ts.now()
        geocentric = sat.at(t)
        x, y, z = geocentric.frame_xyz(itrs).km

        sat_scatter._offsets3d = ([x], [y], [z])
        sat_line.set_data([0, x], [0, y])
        sat_line.set_3d_properties([0, z])

        subpoint = wgs84.subpoint(geocentric)
        title = (
            f"{sat.name}  "
            f"lat {subpoint.latitude.degrees:+.2f}°  "
            f"lon {subpoint.longitude.degrees:+.2f}°  "
            f"alt {subpoint.elevation.km:.1f} km"
        )
        ax.set_title(title, color="white")
        return sat_scatter, sat_line

    anim = animation.FuncAnimation(
        fig,
        update,
        interval=int(args.update_seconds * 1000),
        blit=False,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
