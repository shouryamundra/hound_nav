#!/usr/bin/env python3
"""Dora controller for the holonomic planner: inputs map + track, output geometry_msgs/Twist (body vx, vy, wz).

Replaces controller_dora_node.py (Ackermann) when the planner node_type is ``holonomic_6D``. The robot's
HolonomicLlController subscribes to this Twist.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HOUND_NAV_ROOT = Path(__file__).resolve().parent.parent
if _HOUND_NAV_ROOT.is_dir():
    sys.path.insert(0, str(_HOUND_NAV_ROOT))

import traceback

import rclpy
from dora import Node
from geometry_msgs.msg import Twist
from rclpy.node import Node as RosNode
from rclpy.qos import qos_profile_sensor_data

from hound_nav.holonomic_mppi import HolonomicMPPI
from hound_nav.nav_config import load_hound_nav_config, stack_config
from hound_nav.pdef_codec import unpack_map, unpack_track


def main() -> None:
    launch_cfg = load_hound_nav_config()
    cfg = stack_config(launch_cfg)
    rclpy.init()
    ros = RosNode("hound_nav_holonomic_controller")
    cmd_pub = ros.create_publisher(
        Twist, str(launch_cfg.get("cmd_topic", "/hound_nav/cmd_ackermann")), qos_profile_sensor_data
    )
    dora = Node()
    mppi = HolonomicMPPI(
        cfg["MPPI_config"], cfg["Dynamics_config"], cfg["Sampling_config"], cfg["Cost_config"], cfg["Map_config"]
    )

    def publish(vx: float, vy: float, wz: float) -> None:
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = float(vx), float(vy), float(wz)
        cmd_pub.publish(cmd)

    map_ready = False
    for event in dora:
        if event is None or event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT":
            continue
        try:
            if event.get("id") == "map":
                m = unpack_map(event.get("value"), event.get("metadata") or {})
                mppi.set_costmap(m["costmap"], m["map_center"])
                map_ready = True
            elif event.get("id") == "track" and map_ready:
                tr = unpack_track(event.get("value"), event.get("metadata") or {})
                s, ref = tr["state"], tr["reference"].copy()        # state[5] yaw, [6:8] body v, [14] yaw rate
                if tr["goal_reached"] or len(ref) < 1:
                    mppi.reset()
                    publish(0.0, 0.0, 0.0)
                    continue
                ref[:, :2] += s[:2]                                  # reference is robot-relative in xy
                publish(*mppi.step([s[0], s[1], s[5], s[6], s[7], s[14]], ref))
        except Exception:
            traceback.print_exc()

    ros.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
