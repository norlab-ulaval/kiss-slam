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
    pc2_to_numpy,
)

from pyquaternion import Quaternion

from kiss_slam.config import load_config
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty
from tf2_ros import TransformBroadcaster

from kiss_slam.slam import KissSLAM
import time


class KissSLAMNode(Node):
    def __init__(self):
        """Create parameters, subscriptions, publishers, and set up SLAM"""
        super().__init__("kiss_slam_node")

        # Params specific to ROS wrapper
        self.declare_parameter("points_topic", "/points")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("final_map_file_name", "kiss_slam_map.csv")
        self.declare_parameter("final_trajectory_file_name", "kiss_slam_trajectory.tum")
        self.declare_parameter("final_logs_file_name", "kiss_slam_results.log")

        self.points_topic = self.get_parameter("points_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.final_map_file_name = self.get_parameter("final_map_file_name").value
        self.final_trajectory_file_name = self.get_parameter("final_trajectory_file_name").value
        self.final_logs_file_name = self.get_parameter("final_logs_file_name").value

        # Subscribers
        self.cloud_sub = self.create_subscription(PointCloud2, self.points_topic, self.cloud_cb, 10)

        # Publishers
        self.tf_broadcaster = TransformBroadcaster(self)

        self.kiss_slam_config = load_config(None)
        self.slam = KissSLAM(self.kiss_slam_config)
        self.runtimes = []
        self.timestamps = []

    def cloud_cb(self, in_msg: PointCloud2):
        time_start = time.time()

        # Store timestamp from message
        timestamp = in_msg.header.stamp.sec + in_msg.header.stamp.nanosec * 1e-9
        self.timestamps.append(timestamp)

        pcd = pc2_to_numpy(in_msg)
        self.slam.process_scan(pcd, np.empty((0,)))

        self.runtimes.append(time.time() - time_start)

    def done_cb(self):
        # Check if any scans were processed
        if len(self.timestamps) == 0:
            print("[kiss_slam_node] No scans processed, skipping finalization")
            return

        # Finalize SLAM like in pipeline
        self.slam.generate_new_node()
        self.slam.local_map_graph.erase_last_local_map()
        self.poses, self.pose_graph = self.slam.fine_grained_optimization()
        self.poses = np.array(self.poses)

        # Save results
        self.save_map()
        self.save_trajectory()
        self.save_logs()

    def save_map(self):
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

        global_map = np.vstack(all_points)
        np.savetxt(
            self.final_map_file_name,
            global_map[:, :3],
            fmt="%.4f",
            delimiter=",",
            header="x,y,z",
            comments="",
        )

    def save_trajectory(self):
        # Export results
        with open(self.final_trajectory_file_name, "w") as f:
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

    def save_logs(self):
        with open(self.final_logs_file_name, "w") as f:
            total_time = sum(self.runtimes)
            f.write(f"Total processing time: {total_time:.4f} seconds\n")
            f.write(f"Number of scans processed: {len(self.runtimes)}\n")
            f.write(f"Average runtime: {total_time / len(self.runtimes):.4f} seconds\n")
            f.write(f"Average frequency: {len(self.runtimes) / total_time:.4f} Hz\n")
            f.write(f"Number of closures found: {len(self.slam.closures)}\n")


def main(args=None):
    rclpy.init(args=args)

    kiss = KissSLAMNode()

    try:
        rclpy.spin(kiss)
    except KeyboardInterrupt:
        kiss.done_cb()
    finally:
        kiss.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
