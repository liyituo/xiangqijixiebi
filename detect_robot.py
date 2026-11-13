# -*- coding: utf-8 -*-
"""Read the robot identity and state through the SDK without issuing actuator commands."""

import argparse
import contextlib
import io
import json
import math
import socket
import time
from importlib.metadata import version


def valid_pose(values):
    return (
        isinstance(values, (list, tuple))
        and len(values) == 6
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        )
    )


def detect_robot(ip='192.168.0.232', port=4500, timeout=3.0):
    from pymycobot import Pro450Client

    original_connect = Pro450Client.connect_socket

    def bounded_connection(client):
        return socket.create_connection((client.SERVER_IP, client.SERVER_PORT), timeout=timeout)

    robot = None
    report = {'ip': ip, 'port': port, 'mode': 'read_only', 'detected': False}
    try:
        Pro450Client.connect_socket = bounded_connection
        robot = Pro450Client(ip, port, debug=False)
        Pro450Client.connect_socket = original_connect
        report['sdk_version'] = version('pymycobot')
        time.sleep(0.2)
        report['samples'] = []
        for _ in range(2):
            angles = robot.get_angles()
            coords = robot.get_coords()
            report['samples'].append({
                'angles_deg': angles,
                'coords_mm_deg': coords,
                'valid': valid_pose(angles) and valid_pose(coords),
            })
            time.sleep(0.1)
        for name in ['get_system_version', 'get_modified_version', 'get_atom_version',
                     'is_power_on', 'is_moving', 'get_free_move_mode',
                     'get_error_information', 'get_servo_status']:
            try:
                report[name] = getattr(robot, name)()
            except Exception as error:
                report[name] = {'error': str(error)}
        sdk_messages = io.StringIO()
        with contextlib.redirect_stdout(sdk_messages):
            report['robot_status'] = robot.get_robot_status()
        report['sdk_messages'] = sdk_messages.getvalue().splitlines()
        report['detected'] = all(sample['valid'] for sample in report['samples'])
    except Exception as error:
        report['error'] = str(error)
    finally:
        Pro450Client.connect_socket = original_connect
        if robot is not None:
            robot.close()
    return report


def main():
    parser = argparse.ArgumentParser(description='只读检测 Pro450 连接、位姿与状态')
    parser.add_argument('--ip', default='192.168.0.232')
    parser.add_argument('--port', type=int, default=4500)
    parser.add_argument('--timeout', type=float, default=3.0)
    args = parser.parse_args()
    report = detect_robot(args.ip, args.port, args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['detected'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
