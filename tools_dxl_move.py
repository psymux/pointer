#!/usr/bin/env python3
import argparse
import sys
import time

from dynamixel_sdk import PortHandler, PacketHandler


ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

OPERATING_MODE_POSITION = 3
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


def main():
    parser = argparse.ArgumentParser(description="Move a DYNAMIXEL to a target position.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial device path.")
    parser.add_argument("--id", type=int, default=1, help="DYNAMIXEL ID.")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate.")
    parser.add_argument("--position", type=int, default=2048, help="Goal position (0-4095).")
    parser.add_argument("--wait", type=float, default=1.0, help="Seconds to wait before reading position.")
    parser.add_argument("--disable-torque", action="store_true", help="Disable torque before exit.")
    args = parser.parse_args()

    port_handler = PortHandler(args.port)
    packet_handler = PacketHandler(2.0)

    if not port_handler.openPort():
        print(f"Failed to open port {args.port}", file=sys.stderr)
        return 1
    if not port_handler.setBaudRate(args.baud):
        print(f"Failed to set baud {args.baud}", file=sys.stderr)
        return 1

    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
        port_handler, args.id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION
    )
    if dxl_comm_result != 0 or dxl_error != 0:
        print("Failed to set operating mode.")

    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
        port_handler, args.id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
    )
    if dxl_comm_result != 0 or dxl_error != 0:
        print("Failed to enable torque.")
        port_handler.closePort()
        return 1

    dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(
        port_handler, args.id, ADDR_GOAL_POSITION, int(args.position)
    )
    if dxl_comm_result != 0 or dxl_error != 0:
        print("Failed to write goal position.")
        port_handler.closePort()
        return 1

    time.sleep(args.wait)

    present_pos, dxl_comm_result, dxl_error = packet_handler.read4ByteTxRx(
        port_handler, args.id, ADDR_PRESENT_POSITION
    )
    if dxl_comm_result == 0 and dxl_error == 0:
        print(f"Present position: {present_pos}")
    else:
        print("Failed to read present position.")

    if args.disable_torque:
        packet_handler.write1ByteTxRx(
            port_handler, args.id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )

    port_handler.closePort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
