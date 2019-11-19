from baseline_racer import BaselineRacer
from gtp_visualize import *
from utils import to_airsim_vector, to_airsim_vectors
import airsimneurips as airsim
import argparse
import gtp
import numpy as np
import time

def main(args):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl()
    client.arm()

    traj_tracker_gains = airsim.TrajectoryTrackerGains(
        kp_cross_track = 5.0, kd_cross_track = 0.0,
        kp_vel_cross_track = 3.0, kd_vel_cross_track = 0.0,
        kp_along_track = 0.4, kd_along_track = 0.0,
        kp_vel_along_track = 0.04, kd_vel_along_track = 0.0,
        kp_z_track = 2.0, kd_z_track = 0.0,
        kp_vel_z = 0.4, kd_vel_z = 0.0,
        kp_yaw = 3.0, kd_yaw = 0.1
        )

    client.setTrajectoryTrackerGains(traj_tracker_gains)
    client.takeoffAsync().join()
