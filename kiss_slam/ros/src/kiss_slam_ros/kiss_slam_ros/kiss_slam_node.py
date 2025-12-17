# MIT License
#
# Copyright (c) 2025 Nathan Hewitt, Tiziano Guadagnino, Benedikt Mersch,
# Saurabh Gupta, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os

import numpy as np
import rclpy
from kiss_slam_ros.conversions import (
    build_map,
    build_odometry,
    build_transform,
    matrix_to_pose,
    pc2_to_numpy,
)

from pyquaternion import Quaternion

from geometry_msgs.msg import PoseStamped
from sensor_msgs_py import point_cloud2

from kiss_slam.config import load_config
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty, Header
from tf2_ros import TransformBroadcaster

from kiss_slam.slam import KissSLAM


class KissSLAMNode(Node):
    def __init__(self):
        """Create parameters, subscriptions, publishers, and set up SLAM"""
        super().__init__("kiss_slam_node")

        # Params specific to ROS wrapper
        self.declare_parameter("points_topic", "/points")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("position_covariance", 0.1)
        self.declare_parameter("orientation_covariance", 0.1)

        self.points_topic = self.get_parameter("points_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.position_covariance = self.get_parameter("position_covariance").value
        self.orientation_covariance = self.get_parameter("orientation_covariance").value

        # Subscribers
        self.cloud_sub = self.create_subscription(PointCloud2, self.points_topic, self.cloud_cb, 10)

        self.done_sub = self.create_subscription(Empty, "done", self.done_cb, 10)

        # Publishers
        self.odom_publisher = self.create_publisher(Odometry, "odom", 10)
        self.map_publisher = self.create_publisher(PointCloud2, "map", 10)
        self.pose_publisher = self.create_publisher(PoseStamped, "estimated_pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.kiss_slam_config = load_config(None)
        self.slam = KissSLAM(self.kiss_slam_config)
        self.freqs = []
        self.timestamps = []

    def publish_map(self, points: np.ndarray):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "/map"
        map_msg = point_cloud2.create_cloud_xyz32(header, points[:, :3])

        self.map_publisher.publish(map_msg)

    def publish_pose(self, pose: PoseStamped):
        self.pose_publisher.publish(pose)

    def cloud_cb(self, in_msg: PointCloud2):
        start = self.get_clock().now()
        # Store timestamp from message
        timestamp = in_msg.header.stamp.sec + in_msg.header.stamp.nanosec * 1e-9
        self.timestamps.append(timestamp)

        pcd = pc2_to_numpy(in_msg)
        self.slam.process_scan(pcd, np.empty((0,)))

        pose = matrix_to_pose(self.slam.poses[-1])
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "/map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        self.publish_pose(pose_stamped)

        header = Header()
        header.stamp = in_msg.header.stamp
        header.frame_id = self.map_frame

        # t = build_transform(header, pose, in_msg.header.frame_id)
        # self.tf_broadcaster.sendTransform(t)

        # odom = build_odometry(header, pose, self.position_covariance, self.orientation_covariance)
        # self.odom_publisher.publish(odom)

        self.publish_map(self.slam.voxel_grid.point_cloud())

        duration = self.get_clock().now() - start
        elapsed_s = duration.nanoseconds * 1e-9
        if elapsed_s > 0:
            freq = 1.0 / elapsed_s
            self.freqs.append(freq)

    def done_cb(self, _: Empty):
        self.slam.generate_new_node()
        self.slam.local_map_graph.erase_last_local_map()
        self.poses, self.pose_graph = self.slam.fine_grained_optimization()
        self.poses = np.array(self.poses)

        # Export poses to TUM format (matching pipeline implementation)
        path = "/home/nicolas-lauzon"
        tum_file = os.path.join(path, "trajectory_red.tum")

        with open(tum_file, "w") as f:
            for i, pose_matrix in enumerate(self.poses):
                # Extract translation
                tx, ty, tz = pose_matrix[:3, -1].flatten()

                # Extract rotation and convert to quaternion (w, x, y, z) order
                qw, qx, qy, qz = Quaternion(matrix=pose_matrix, atol=0.01).elements

                # Use actual timestamp from message
                timestamp = self.timestamps[i]

                # Write in TUM format: timestamp tx ty tz qx qy qz qw
                f.write(
                    f"{timestamp:.4f} {tx:.4f} {ty:.4f} {tz:.4f} {qx:.4f} {qy:.4f} {qz:.4f} {qw:.4f}\n"
                )

        # Save map (point cloud) to CSV
        # Create global map by transforming all local maps to global frame
        all_points = []
        keyposes = self.slam.get_keyposes()

        for i, local_map in enumerate(self.slam.local_map_graph.local_maps()):
            if i >= len(keyposes):
                break

            # Get the local map point cloud
            local_pcd = local_map.pcd.point.positions.cpu().numpy()

            # Transform to global frame using the keypose
            R = keyposes[i][:3, :3]
            t = keyposes[i][:3, -1]
            global_points = local_pcd @ R.T + t

            all_points.append(global_points)

        # Combine all points
        if all_points:
            global_map = np.vstack(all_points)
        else:
            global_map = np.array([]).reshape(0, 3)

        map_file = os.path.join(path, "map_red.csv")
        np.savetxt(
            map_file, global_map[:, :3], fmt="%.4f", delimiter=",", header="x,y,z", comments=""
        )

        log_file = os.path.join(path, "result_metrics_red.log")
        with open(log_file, "w") as f:
            f.write("────────────────── Results ────────────────────\n")
            f.write("+--------------------------+-------+----------+\n")
            f.write("|                   Metric | Value | Units    |\n")
            f.write("+==========================+=======+==========+\n")
            f.write(f"|          Average freq |  {np.mean(self.freqs):.0f}   | Hz       |\n")
            f.write(
                f"| Number of closures found |   {len(self.slam.get_closures())}   | closures |\n"
            )
            f.write("+--------------------------+-------+----------+\n")

        self.get_logger().info(f"Exported {len(self.poses)} poses to {tum_file}")


def main(args=None):
    rclpy.init(args=args)

    kiss = KissSLAMNode()

    rclpy.spin(kiss)

    kiss.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
