#!/usr/bin/env python3
import argparse
import glob
import os
import sys

from dynamixel_sdk import PortHandler, PacketHandler


COMMON_BAUDS = [9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000]


def find_default_port():
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    if by_id:
        return by_id[0]
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def scan_port(port, bauds, ids, protocol=2.0):
    port_handler = PortHandler(port)
    packet_handler = PacketHandler(protocol)

    if not port_handler.openPort():
        print(f"Failed to open port {port}", file=sys.stderr)
        return []

    found = []
    for baud in bauds:
        if not port_handler.setBaudRate(baud):
            continue
        for dxl_id in ids:
            dxl_model, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)
            if dxl_comm_result == 0 and dxl_error == 0:
                found.append((dxl_id, baud, dxl_model))
                print(f"Found ID {dxl_id} at {baud} bps (model {dxl_model})")

    port_handler.closePort()
    return found


def main():
    parser = argparse.ArgumentParser(description="Scan for DYNAMIXEL devices.")
    parser.add_argument("--port", default=None, help="Serial device path (e.g., /dev/ttyACM0).")
    parser.add_argument("--ids", default="0-2", help="ID list or range, e.g. 0-5 or 1,2,3.")
    parser.add_argument("--bauds", default=None, help="Comma list of bauds to scan.")
    args = parser.parse_args()

    port = args.port or find_default_port()
    if not port:
        print("No serial port found.", file=sys.stderr)
        return 1

    if args.bauds:
        bauds = [int(x) for x in args.bauds.split(",") if x.strip()]
    else:
        bauds = COMMON_BAUDS

    if "-" in args.ids:
        start, end = args.ids.split("-", 1)
        ids = list(range(int(start), int(end) + 1))
    else:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]

    print(f"Scanning port {port}")
    found = scan_port(port, bauds, ids)
    if not found:
        print("No devices found.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
