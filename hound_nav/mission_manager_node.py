"""GPS / RViz mission manager → one PoseStamped on goal_topic + Path viz.

GPS mode
--------
Vehicle pose comes from NavSatFix + IMU (heading). A new GPS waypoint set
(nav_msgs/Path, frame wgs84, x=lat y=lon z=alt) is accepted only when
MAVLink fix_type >= min_fix_type (3 = 3D) and horizontal accuracy < max_h_acc_m.

Local ENU is east/north about a GPS origin frozen at accept (unless
update_wp_local_from_gps). Goal pose heading is always
atan2(target_y - robot_y, target_x - robot_x) in that frame.

If control_state is available, the GPS ENU delta is applied at the EKF
robot xy so the planner's odom frame matches. Current target goes to
goal_topic; the full local-frame list is latched on path_viz_topic for RViz.

RViz mode
---------
Clicks from RViz 2D Goal Pose (PoseStamped, typically frame map) are stored
as map-frame waypoints, written to mission_file YAML, and latched as a Path.
While record is true, goal_topic is not republished (avoids echoing the click
back into the list). With record false and a loaded file, the current
waypoint is published on goal_topic.
"""

from __future__ import annotations

import math
import signal
from pathlib import Path as FsPath
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import Empty, Float64MultiArray, UInt8
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
import tf2_geometry_msgs  # noqa: F401  — registers PoseStamped for Buffer.transform

from hound_nav.path_bezier import carrot_on_path, interpolate_path

_EARTH_RADIUS_M = 6378137.0


def wgs84_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> np.ndarray:
    dlat = math.radians(lat_deg - lat0_deg)
    dlon = math.radians(lon_deg - lon0_deg)
    cos_lat0 = math.cos(math.radians(lat0_deg))
    north = dlat * _EARTH_RADIUS_M
    east = dlon * _EARTH_RADIUS_M * cos_lat0
    up = alt_m - alt0_m
    return np.array([east, north, up], dtype=np.float64)


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class MissionManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_manager")

        self.declare_parameter("mode", "gps")
        self.declare_parameter("gps_topic", "/hound_fcu_control/gps/fix")
        self.declare_parameter("gps_fix_type_topic", "/hound_fcu_control/gps/fix_type")
        self.declare_parameter("imu_topic", "/hound_fcu_control/imu")
        self.declare_parameter(
            "gps_waypoints_topic", "/hound_fcu_control/mission/gps"
        )
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("path_viz_topic", "/hound_nav/mission/waypoints")
        self.declare_parameter("state_topic", "/hound_fcu_control/control_state")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("goal_frame_id", "odom")
        self.declare_parameter("min_fix_type", 3)
        self.declare_parameter("max_h_acc_m", 2.5)
        self.declare_parameter("update_wp_local_from_gps", False)
        self.declare_parameter("wp_radius", 1.0)
        # Periodic goal republish; 0 disables the timer (still publishes on WP reach / mission accept).
        self.declare_parameter("publish_hz", 0.25)
        self.declare_parameter("mission_file", "")
        self.declare_parameter("record", True)
        self.declare_parameter("load_on_start", False)
        self.declare_parameter("save_on_shutdown", True)
        self.declare_parameter("publish_goals", False)
        self.declare_parameter("min_separation_m", 0.3)
        self.declare_parameter("waypoint_topic", "/goal_pose")
        self.declare_parameter("path_interp", "none")  # none | bezier
        self.declare_parameter("bezier_samples", 40)
        self.declare_parameter("lookahead", 8.0)

        self._mode = str(self.get_parameter("mode").value).strip().lower()
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._goal_frame_id = str(self.get_parameter("goal_frame_id").value)
        self._min_fix = int(self.get_parameter("min_fix_type").value)
        self._max_h_acc = float(self.get_parameter("max_h_acc_m").value)
        self._update_local = bool(self.get_parameter("update_wp_local_from_gps").value)
        self._wp_radius = float(self.get_parameter("wp_radius").value)
        self._path_interp = str(self.get_parameter("path_interp").value).strip().lower()
        self._bezier_samples = int(self.get_parameter("bezier_samples").value)
        self._lookahead = float(self.get_parameter("lookahead").value)

        goal_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        gps_wp_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._goal_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("goal_topic").value), goal_qos
        )
        self._path_viz_pub = self.create_publisher(
            Path, str(self.get_parameter("path_viz_topic").value), goal_qos
        )

        self._fix: Optional[NavSatFix] = None
        self._fix_type: int = 0
        self._have_fix_type = False
        self._imu_yaw: Optional[float] = None
        self._ekf_xy: Optional[np.ndarray] = None

        self._pending_gps: Optional[np.ndarray] = None  # Nx3 lat,lon,alt
        self._gps_wps: Optional[np.ndarray] = None
        self._wp_local: Optional[np.ndarray] = None
        self._origin: Optional[np.ndarray] = None  # lat, lon, alt
        self._curr_wp_index = 0
        self._final_reached = False
        self._logged_rviz = False
        self._last_reject_log = 0.0
        self._rviz_poses: List[PoseStamped] = []
        self._rviz_dirty = False
        self._rviz_record = bool(self.get_parameter("record").value)
        self._rviz_publish_goals = bool(self.get_parameter("publish_goals").value)
        self._rviz_min_sep = float(self.get_parameter("min_separation_m").value)
        self._mission_file = str(self.get_parameter("mission_file").value).strip()
        self._save_on_shutdown = bool(self.get_parameter("save_on_shutdown").value)

        if self._mode == "rviz":
            self._init_rviz_mode()
            return
        if self._mode != "gps":
            self.get_logger().error(f"unknown mode={self._mode!r} (use gps or rviz)")
            return

        self.create_subscription(
            NavSatFix,
            str(self.get_parameter("gps_topic").value),
            self._on_gps,
            10,
        )
        self.create_subscription(
            UInt8,
            str(self.get_parameter("gps_fix_type_topic").value),
            self._on_fix_type,
            10,
        )
        self.create_subscription(
            Imu, str(self.get_parameter("imu_topic").value), self._on_imu, 10
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("gps_waypoints_topic").value),
            self._on_gps_waypoints,
            gps_wp_qos,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("state_topic").value),
            self._on_state,
            10,
        )
        hz = float(self.get_parameter("publish_hz").value)
        if hz > 0.0:
            self.create_timer(1.0 / hz, self._on_timer)

        self.get_logger().info(
            f"GPS mission manager: wps={self.get_parameter('gps_waypoints_topic').value} "
            f"→ {self.get_parameter('goal_topic').value} "
            f"viz={self.get_parameter('path_viz_topic').value} "
            f"min_fix={self._min_fix} max_h_acc={self._max_h_acc:.2f}m "
            f"update_local={self._update_local} publish_hz={hz:g} "
            f"interp={self._path_interp} lookahead={self._lookahead:.1f}m "
            "(also publishes on waypoint reach)"
        )

    def _on_fix_type(self, msg: UInt8) -> None:
        self._fix_type = int(msg.data)
        self._have_fix_type = True
        self._try_activate()

    def _on_gps(self, msg: NavSatFix) -> None:
        self._fix = msg
        if not self._have_fix_type:
            # FCU maps MAVLink >=3 → STATUS_FIX; treat as 3D if topic is late.
            if msg.status.status >= NavSatStatus.STATUS_FIX:
                self._fix_type = max(self._fix_type, self._min_fix)
        if self._update_local and self._gps_wps is not None and self._gps_ok():
            self._origin = np.array(
                [msg.latitude, msg.longitude, msg.altitude], dtype=np.float64
            )
            self._rebuild_local()
            self._publish_waypoints()
        self._try_activate()
        if self._maybe_advance():
            self._publish_goal()

    def _on_imu(self, msg: Imu) -> None:
        self._imu_yaw = yaw_from_quat(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )

    def _on_state(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self._ekf_xy = np.array([msg.data[0], msg.data[1]], dtype=np.float64)
        if self._mode == "rviz" and self._maybe_advance_rviz():
            if self._rviz_publish_goals and not self._rviz_record:
                self._publish_rviz_goal()

    def _on_gps_waypoints(self, msg: Path) -> None:
        if not msg.poses:
            return
        rows: List[List[float]] = []
        for ps in msg.poses:
            rows.append(
                [ps.pose.position.x, ps.pose.position.y, ps.pose.position.z]
            )
        self._pending_gps = np.array(rows, dtype=np.float64)
        self.get_logger().info(
            f"GPS mission received n={len(rows)} frame={msg.header.frame_id!r}"
        )
        self._try_activate()

    def _h_acc_m(self) -> float:
        if self._fix is None:
            return float("inf")
        cov0 = float(self._fix.position_covariance[0])
        if cov0 > 0.0 and math.isfinite(cov0):
            return math.sqrt(cov0)
        return float("inf")

    def _gps_ok(self) -> bool:
        if self._fix is None:
            return False
        if self._fix_type < self._min_fix:
            return False
        return self._h_acc_m() < self._max_h_acc

    def _try_activate(self) -> None:
        if self._pending_gps is None:
            return
        if not self._gps_ok():
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_reject_log >= 2.0:
                self._last_reject_log = now
                self.get_logger().warn(
                    "holding GPS mission: need fix_type>="
                    f"{self._min_fix} (have {self._fix_type}) and "
                    f"h_acc<{self._max_h_acc:.2f}m (have {self._h_acc_m():.2f}m)"
                )
            return
        assert self._fix is not None
        self._gps_wps = np.copy(self._pending_gps)
        self._pending_gps = None
        self._origin = np.array(
            [self._fix.latitude, self._fix.longitude, self._fix.altitude],
            dtype=np.float64,
        )
        self._curr_wp_index = 0
        self._final_reached = False
        self._rebuild_local()
        self.get_logger().info(
            f"GPS mission active n={int(self._gps_wps.shape[0])} "
            f"origin=({self._origin[0]:.7f},{self._origin[1]:.7f}) "
            f"idx=0 local=({self._wp_local[0, 0]:.2f},{self._wp_local[0, 1]:.2f})"
        )
        self._publish_goal()
        self._publish_waypoints()

    def _rebuild_local(self) -> None:
        if self._gps_wps is None or self._origin is None:
            self._wp_local = None
            return
        n = int(self._gps_wps.shape[0])
        out = np.zeros((n, 3), dtype=np.float64)
        lat0, lon0, alt0 = (float(self._origin[0]), float(self._origin[1]), float(self._origin[2]))
        for i in range(n):
            out[i] = wgs84_to_enu(
                float(self._gps_wps[i, 0]),
                float(self._gps_wps[i, 1]),
                float(self._gps_wps[i, 2]),
                lat0,
                lon0,
                alt0,
            )
        self._wp_local = out

    def _robot_local(self) -> Optional[np.ndarray]:
        if self._fix is None or self._origin is None:
            return None
        return wgs84_to_enu(
            self._fix.latitude,
            self._fix.longitude,
            self._fix.altitude,
            float(self._origin[0]),
            float(self._origin[1]),
            float(self._origin[2]),
        )

    def _maybe_advance(self) -> bool:
        """Return True if a waypoint was reached (index advanced or final WP)."""
        if self._use_bezier():
            return False
        if self._wp_local is None or self._curr_wp_index >= len(self._wp_local):
            return False
        robot = self._robot_local()
        if robot is None:
            return False
        wp = self._wp_local[self._curr_wp_index]
        d = float(np.linalg.norm(wp[:2] - robot[:2]))
        last = int(self._wp_local.shape[0]) - 1
        if d >= self._wp_radius:
            return False
        if self._curr_wp_index < last:
            self._curr_wp_index += 1
            nxt = self._wp_local[self._curr_wp_index]
            self.get_logger().info(
                f"waypoint reached → {self._curr_wp_index}/{last} "
                f"local=({nxt[0]:.2f},{nxt[1]:.2f})"
            )
            return True
        if not self._final_reached:
            self._final_reached = True
            self.get_logger().info(f"final waypoint {last} reached")
            return True
        return False

    def _world_xy(self, wp_xy: np.ndarray) -> Optional[np.ndarray]:
        """Waypoint in goal/odom frame (GPS ENU, plus EKF offset when available)."""
        robot = self._robot_local()
        if robot is None:
            return None
        xy = np.asarray(wp_xy, dtype=np.float64)[:2]
        if self._ekf_xy is not None:
            return self._ekf_xy + (xy - robot[:2])
        return xy.copy()

    def _publish_waypoints(self) -> None:
        if self._wp_local is None or int(self._wp_local.shape[0]) < 1:
            return
        pts = []
        for i in range(int(self._wp_local.shape[0])):
            xy = self._world_xy(self._wp_local[i])
            if xy is None:
                return
            pts.append(xy)
        arr = np.asarray(pts, dtype=np.float64)
        if self._use_bezier() and arr.shape[0] >= 2:
            arr = interpolate_path(arr, self._bezier_samples)
        self._publish_xy_path(arr, self._frame_id)

    def _publish_goal(self) -> None:
        if self._wp_local is None or int(self._wp_local.shape[0]) < 1:
            return
        robot = self._robot_local()
        if robot is None:
            return
        robot_xy = self._world_xy(robot)
        if robot_xy is None:
            return
        pts = []
        for i in range(int(self._wp_local.shape[0])):
            xy = self._world_xy(self._wp_local[i])
            if xy is None:
                return
            pts.append(xy)
        arr = np.asarray(pts, dtype=np.float64)
        if self._use_bezier() and arr.shape[0] >= 2:
            curve = interpolate_path(arr, self._bezier_samples)
            hit = carrot_on_path(curve, robot_xy, self._lookahead, self._wp_radius)
            if hit is None:
                return
            goal_xy, yaw, done = hit
            if done and not self._final_reached:
                self._final_reached = True
                self.get_logger().info("bezier: end of mission")
            self._publish_carrot_goal(goal_xy, yaw, self._frame_id)
            return
        idx = min(self._curr_wp_index, int(self._wp_local.shape[0]) - 1)
        target_xy = arr[idx]
        dx = float(target_xy[0] - robot_xy[0])
        dy = float(target_xy[1] - robot_xy[1])
        if dx * dx + dy * dy < 1.0e-8 and self._imu_yaw is not None:
            yaw = float(self._imu_yaw)
        else:
            yaw = math.atan2(dy, dx)
        self._publish_carrot_goal(target_xy, yaw, self._frame_id)

    def _on_timer(self) -> None:
        if self._mode == "rviz":
            if self._maybe_advance_rviz():
                pass
            self._publish_rviz_path()
            if self._rviz_publish_goals and not self._rviz_record:
                self._publish_rviz_goal()
            return
        if self._mode != "gps":
            return
        self._publish_goal()
        self._publish_waypoints()

    def _init_rviz_mode(self) -> None:
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        wp_topic = str(self.get_parameter("waypoint_topic").value)
        if bool(self.get_parameter("load_on_start").value) and self._mission_file:
            n = self._load_rviz_mission(FsPath(self._mission_file))
            self.get_logger().info(f"loaded {n} waypoints from {self._mission_file}")
        self.create_subscription(PoseStamped, wp_topic, self._on_rviz_waypoint, 10)
        self.create_subscription(Empty, "~/clear", self._on_rviz_clear, 10)
        self.create_subscription(Empty, "~/undo", self._on_rviz_undo, 10)
        self.create_subscription(Empty, "~/advance", self._on_rviz_advance, 10)
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("state_topic").value),
            self._on_state,
            10,
        )
        self.create_service(Trigger, "~/save", self._on_rviz_save_srv)
        hz = float(self.get_parameter("publish_hz").value)
        if hz > 0.0:
            self.create_timer(1.0 / hz, self._on_timer)
        self._publish_rviz_path()
        if self._rviz_publish_goals and not self._rviz_record:
            self._publish_rviz_goal()
        self.get_logger().info(
            f"RViz mission: record={self._rviz_record} "
            f"clicks={wp_topic} store={self._frame_id} "
            f"goal_frame={self._goal_frame_id} "
            f"file={self._mission_file or '(none)'} "
            f"publish_goals={self._rviz_publish_goals} "
            f"wp_radius={self._wp_radius:.2f}m "
            f"interp={self._path_interp} lookahead={self._lookahead:.1f}m "
            f"viz={self.get_parameter('path_viz_topic').value} "
            f"n={len(self._rviz_poses)}"
        )

    def _transform_pose(
        self,
        msg: PoseStamped,
        target: str,
        *,
        what: str,
    ) -> Optional[PoseStamped]:
        src = str(msg.header.frame_id or "").strip()
        if not src:
            self.get_logger().warn(f"{what}: empty frame_id; need TF to {target}")
            return None
        if src == target:
            out = PoseStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = target
            out.pose = msg.pose
            return out
        stamped = PoseStamped()
        stamped.header.frame_id = src
        stamped.header.stamp = Time().to_msg()
        stamped.pose = msg.pose
        try:
            out = self._tf_buffer.transform(
                stamped, target, timeout=Duration(seconds=0.3)
            )
        except TransformException as exc:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_reject_log >= 2.0:
                self._last_reject_log = now
                self.get_logger().warn(f"{what}: no TF {target} ← {src} ({exc})")
            return None
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = target
        return out

    def _use_bezier(self) -> bool:
        return self._path_interp in ("bezier", "bezier_curve", "spline")

    def _robot_xy_in_frame(self, frame: str) -> Optional[np.ndarray]:
        if self._ekf_xy is None:
            return None
        if str(frame) == str(self._goal_frame_id):
            return np.array([float(self._ekf_xy[0]), float(self._ekf_xy[1])], dtype=np.float64)
        ps = PoseStamped()
        ps.header.frame_id = self._goal_frame_id
        ps.pose.position.x = float(self._ekf_xy[0])
        ps.pose.position.y = float(self._ekf_xy[1])
        ps.pose.orientation.w = 1.0
        out = self._transform_pose(ps, frame, what="robot")
        if out is None:
            return None
        return np.array(
            [float(out.pose.position.x), float(out.pose.position.y)], dtype=np.float64
        )

    def _publish_xy_path(self, pts: np.ndarray, frame: str) -> None:
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = frame
        xy = np.asarray(pts, dtype=np.float64)
        n = int(xy.shape[0]) if xy.ndim == 2 else 0
        for i in range(n):
            if i + 1 < n:
                yaw = math.atan2(float(xy[i + 1, 1] - xy[i, 1]), float(xy[i + 1, 0] - xy[i, 0]))
            elif i > 0:
                yaw = math.atan2(float(xy[i, 1] - xy[i - 1, 1]), float(xy[i, 0] - xy[i - 1, 0]))
            else:
                yaw = 0.0
            qx, qy, qz, qw = yaw_to_quat(yaw)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(xy[i, 0])
            ps.pose.position.y = float(xy[i, 1])
            ps.pose.position.z = 0.0
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)
        self._path_viz_pub.publish(path)

    def _publish_carrot_goal(self, goal_xy: np.ndarray, yaw: float, frame: str) -> None:
        qx, qy, qz, qw = yaw_to_quat(yaw)
        src = PoseStamped()
        src.header.stamp = self.get_clock().now().to_msg()
        src.header.frame_id = frame
        src.pose.position.x = float(goal_xy[0])
        src.pose.position.y = float(goal_xy[1])
        src.pose.position.z = 0.0
        src.pose.orientation.x = qx
        src.pose.orientation.y = qy
        src.pose.orientation.z = qz
        src.pose.orientation.w = qw
        msg = self._transform_pose(src, self._goal_frame_id, what="goal")
        if msg is None:
            return
        self._goal_pub.publish(msg)

    def _on_rviz_waypoint(self, msg: PoseStamped) -> None:
        if not self._rviz_record:
            return
        ps = self._transform_pose(msg, self._frame_id, what="click")
        if ps is None:
            return
        if self._rviz_min_sep > 0.0 and self._rviz_poses:
            last = self._rviz_poses[-1].pose.position
            dx = ps.pose.position.x - last.x
            dy = ps.pose.position.y - last.y
            if math.hypot(dx, dy) < self._rviz_min_sep:
                self.get_logger().warn(
                    f"ignored click ({self._rviz_min_sep:.2f} m from last waypoint)"
                )
                return
        self._rviz_poses.append(ps)
        self._rviz_dirty = True
        self._curr_wp_index = len(self._rviz_poses) - 1
        self._publish_rviz_path()
        self.get_logger().info(
            f"waypoint {len(self._rviz_poses)}: "
            f"({ps.pose.position.x:.2f}, {ps.pose.position.y:.2f}) "
            f"frame={ps.header.frame_id}"
        )

    def _on_rviz_clear(self, _msg: Empty) -> None:
        self._rviz_poses.clear()
        self._curr_wp_index = 0
        self._final_reached = False
        self._rviz_dirty = True
        self._publish_rviz_path()
        self.get_logger().info("mission cleared")

    def _on_rviz_undo(self, _msg: Empty) -> None:
        if not self._rviz_poses:
            return
        self._rviz_poses.pop()
        self._curr_wp_index = max(0, len(self._rviz_poses) - 1)
        self._rviz_dirty = True
        self._publish_rviz_path()
        self.get_logger().info(f"undo → {len(self._rviz_poses)} waypoints")

    def _on_rviz_advance(self, _msg: Empty) -> None:
        if not self._rviz_poses:
            return
        last = len(self._rviz_poses) - 1
        if self._curr_wp_index < last:
            self._curr_wp_index += 1
            self.get_logger().info(f"advance → {self._curr_wp_index}/{last}")
            if self._rviz_publish_goals:
                self._publish_rviz_goal()

    def _on_rviz_save_srv(self, _req: Trigger.Request, res: Trigger.Response):
        ok = self.save_rviz_mission()
        res.success = ok
        res.message = self._mission_file if ok else "save failed"
        return res

    def _publish_rviz_path(self) -> None:
        if not self._rviz_poses:
            path = Path()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = self._frame_id
            self._path_viz_pub.publish(path)
            return
        frame = self._rviz_poses[0].header.frame_id or self._frame_id
        arr = np.array(
            [[ps.pose.position.x, ps.pose.position.y] for ps in self._rviz_poses],
            dtype=np.float64,
        )
        if self._use_bezier() and arr.shape[0] >= 2:
            arr = interpolate_path(arr, self._bezier_samples)
        self._publish_xy_path(arr, frame)

    def _maybe_advance_rviz(self) -> bool:
        """Advance when control_state is within wp_radius of the current WP in goal_frame."""
        if self._rviz_record or not self._rviz_poses:
            return False
        if self._ekf_xy is None:
            return False
        if self._final_reached:
            return False
        if self._use_bezier():
            return False
        idx = min(self._curr_wp_index, len(self._rviz_poses) - 1)
        goal = self._transform_pose(
            self._rviz_poses[idx], self._goal_frame_id, what="advance"
        )
        if goal is None:
            return False
        d = math.hypot(
            float(goal.pose.position.x) - float(self._ekf_xy[0]),
            float(goal.pose.position.y) - float(self._ekf_xy[1]),
        )
        last = len(self._rviz_poses) - 1
        if d >= self._wp_radius:
            return False
        if self._curr_wp_index < last:
            self._curr_wp_index += 1
            nxt = self._rviz_poses[self._curr_wp_index].pose.position
            self.get_logger().info(
                f"waypoint reached (d={d:.2f}m) → {self._curr_wp_index}/{last} "
                f"map=({nxt.x:.2f},{nxt.y:.2f})"
            )
            return True
        self._final_reached = True
        self.get_logger().info(f"final waypoint {last} reached (d={d:.2f}m)")
        return True

    def _publish_rviz_goal(self) -> None:
        if not self._rviz_poses:
            return
        if self._use_bezier() and len(self._rviz_poses) >= 2:
            frame = self._rviz_poses[0].header.frame_id or self._frame_id
            arr = np.array(
                [[ps.pose.position.x, ps.pose.position.y] for ps in self._rviz_poses],
                dtype=np.float64,
            )
            curve = interpolate_path(arr, self._bezier_samples)
            robot = self._robot_xy_in_frame(frame)
            if robot is None:
                return
            hit = carrot_on_path(curve, robot, self._lookahead, self._wp_radius)
            if hit is None:
                return
            goal_xy, yaw, done = hit
            if done and not self._final_reached:
                self._final_reached = True
                self.get_logger().info("bezier: end of mission")
            self._publish_carrot_goal(goal_xy, yaw, frame)
            return
        idx = min(self._curr_wp_index, len(self._rviz_poses) - 1)
        src = self._rviz_poses[idx]
        msg = self._transform_pose(src, self._goal_frame_id, what="goal")
        if msg is None:
            return
        self._goal_pub.publish(msg)

    def _load_rviz_mission(self, path: FsPath) -> int:
        if not path.is_file():
            self.get_logger().error(f"mission file not found: {path}")
            return 0
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        frame = str(data.get("frame_id", self._frame_id))
        self._frame_id = frame
        self._rviz_poses.clear()
        for wp in data.get("waypoints") or []:
            ps = PoseStamped()
            ps.header.frame_id = frame
            if isinstance(wp, dict):
                ps.pose.position.x = float(wp.get("x", 0.0))
                ps.pose.position.y = float(wp.get("y", 0.0))
                ps.pose.position.z = float(wp.get("z", 0.0))
                yaw = float(wp.get("yaw", 0.0))
                qx, qy, qz, qw = yaw_to_quat(yaw)
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                ps.pose.position.x = float(wp[0])
                ps.pose.position.y = float(wp[1])
                ps.pose.position.z = float(wp[2]) if len(wp) > 2 else 0.0
                ps.pose.orientation.w = 1.0
            else:
                continue
            self._rviz_poses.append(ps)
        self._curr_wp_index = 0
        self._final_reached = False
        self._rviz_dirty = False
        return len(self._rviz_poses)

    def save_rviz_mission(self, path: Optional[FsPath] = None) -> bool:
        out = path or (FsPath(self._mission_file) if self._mission_file else None)
        if out is None:
            self.get_logger().warn("no mission_file set — not saving")
            return False
        if not self._rviz_poses:
            self.get_logger().warn("empty mission — not saving")
            return False
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frame_id": self._frame_id,
            "waypoints": [
                {
                    "x": float(p.pose.position.x),
                    "y": float(p.pose.position.y),
                    "z": float(p.pose.position.z),
                    "yaw": yaw_from_quat(
                        p.pose.orientation.x,
                        p.pose.orientation.y,
                        p.pose.orientation.z,
                        p.pose.orientation.w,
                    ),
                }
                for p in self._rviz_poses
            ],
        }
        out.write_text(
            yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        self._rviz_dirty = False
        self.get_logger().info(f"saved {len(self._rviz_poses)} waypoints → {out}")
        return True

    def shutdown_save(self) -> None:
        if (
            self._mode == "rviz"
            and self._save_on_shutdown
            and self._rviz_record
            and self._rviz_dirty
            and self._rviz_poses
        ):
            self.save_rviz_mission()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = MissionManagerNode()
    shutting_down = {"done": False}

    def _handle_sig(_signum, _frame) -> None:
        if shutting_down["done"]:
            return
        shutting_down["done"] = True
        node.shutdown_save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not shutting_down["done"]:
            shutting_down["done"] = True
            try:
                node.shutdown_save()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
