from baseline_racer import BaselineRacer
from gtp_visualize import *
from utils import to_airsim_vector, to_airsim_vectors
import airsimneurips as airsim
import threading
import argparse
import gtp
import numpy as np
import time
# Use non interactive matplotlib backend
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D #3d plotting

# Julia interface to JuMP / Ipopt
# print("Loading Julia...\n")
# import julia
# from julia import Main
# from julia import DroneRacing as dr
dr = None

class Trajectory():
    def __init__(self,pos,vel,accel,t_vec):
        self.pos = pos
        self.vel = vel
        self.accel = accel
        self.t_vec = t_vec


class GlobalTrajectoryOptimizer():
    def __init__(self,traj_params, drone_params, gate_poses, gate_inner_dims, gate_outer_dims):
        self.traj_params = traj_params
        self.drone_params = drone_params
        self.gate_poses = gate_poses
        self.gate_inner_dims = gate_inner_dims
        self.gate_outer_dims = gate_outer_dims
        if self.traj_params.load_traj_from_file:
            pass
        else:
            print("Loading Julia...\n")
            import julia
            from julia import Main
            from julia import DroneRacing
            global dr
            dr = DroneRacing
            dr.warmup()

    def quat_to_julia_vec(self,quat):
        vec = [
            quat.w_val,
            quat.z_val,
            quat.x_val,
            quat.y_val
        ]
        return vec

    def pos_to_julia_vec(self,pos):
        vec = [
            pos.x_val,
            pos.y_val,
            pos.z_val
        ]
        return vec

    def compute_global_optimal_trajectory(self,start_state,
        gate_idxs=[],**kwargs):
        if self.traj_params.load_traj_from_file:
            pos = np.load("pos.npy")
            vel = np.load("vel.npy")
            accel = np.load("accel.npy")
            t_vec = np.load("t_vec.npy")
            return Trajectory(pos,vel,accel,t_vec)
        # Get current state
        start_pos = start_state.kinematics_estimated.position
        start_vel = start_state.kinematics_estimated.linear_velocity
        ####################### Pass information to Julia ######################
        traj_opt_model = dr.VariableTimeParticleTrajOptModel(
            start_pos = [start_pos.x_val, start_pos.y_val, start_pos.z_val],
            start_vel = [start_vel.x_val, start_vel.y_val, start_vel.z_val],
            MAX_VEL = self.traj_params.v_max,
            MAX_ACCEL = self.traj_params.a_max,
            n = self.traj_params.n,
            dt_min = self.traj_params.dt_min,
            dt = self.traj_params.dt,
            r_safe = self.traj_params.r_safe,
            **kwargs # n, dt_min, t0, ...
        )
        if len(gate_idxs) == 0:
            gate_idxs = range(len(self.gate_poses))
        for gate_idx in gate_idxs:
            pose = self.gate_poses[gate_idx]
            pos = self.pos_to_julia_vec(pose.position)
            orientation = self.quat_to_julia_vec(pose.orientation)

            inner_width = [
                self.gate_inner_dims.y_val, # rearrange because gates point along y-direction
                self.gate_inner_dims.x_val,
                self.gate_inner_dims.z_val]
            dr.add_gate(traj_opt_model,dr.Gate3D(pos,orientation,inner_width))
        ########################################################################
        ######################### Optimize Ipopt model #########################
        JuMP_model = dr.formulate_global_traj_opt_problem(traj_opt_model)
        dr.optimize_trajectory(traj_opt_model,JuMP_model)
        ########################################################################
        if self.traj_params.resample:
            # resample with a constant timestep
            pos,vel,accel,t_vec = dr.resample_traj_exact(traj_opt_model,JuMP_model)
        else:
            # leave with a variable time step
            pos = dr.get_x(traj_opt_model,JuMP_model)
            vel = dr.get_v(traj_opt_model,JuMP_model)
            accel = dr.get_a(traj_opt_model,JuMP_model)
            t_vec = dr.get_t_vec(traj_opt_model,JuMP_model)
        # Save
        np.save("pos.npy",pos)
        np.save("vel.npy",vel)
        np.save("accel.npy",accel)
        np.save("t_vec.npy",t_vec)
        return Trajectory(pos,vel,accel,t_vec)

    def cache_trajectory(self):
        filename = "pos.npy"

        pass

class HumDrumRacer(BaselineRacer):
    def __init__(self, traj_params, drone_names, drone_i, drone_params,
             use_vel_constraints=False
             ):
        super().__init__(drone_name=drone_names[drone_i], viz_traj=True)
        self.drone_names = drone_names
        self.drone_i = drone_i
        self.drone_params = drone_params
        self.traj_params = traj_params
        self.use_vel_constraints = use_vel_constraints
        self.planning_cycles = 0
        self.is_replanning_thread_active = False
        self.replanning_callback_thread = threading.Thread(
            target=self.repeat_timer_replanning_callback,
            args=(self.replanning_callback, self.traj_params.dt_planner)
            )

        self.load_level(args.level_name)
        self.get_ground_truth_gate_poses()
        self.global_traj_optimizer = GlobalTrajectoryOptimizer(
            traj_params,
            drone_params[drone_i],
            self.gate_poses_ground_truth,
            self.gate_inner_dims,
            self.gate_outer_dims
            )

        TAKEOFF_SHIFT = -1

        start_state = self.airsim_client.getMultirotorState()
        start_state.kinematics_estimated.position.z_val += TAKEOFF_SHIFT
        self.traj = self.global_traj_optimizer.compute_global_optimal_trajectory(start_state)
        self.step = 0

        velocity_gains = airsim.VelocityControllerGains(
            xGains = airsim.PIDGains(2.0, 0.2, 2.0), # default: 0.2, 0.0, 0.0
            yGains = airsim.PIDGains(2.0, 0.2, 2.0), # default: 0.2, 0.0, 0.0
            zGains = airsim.PIDGains(2.0, 0.2, 2.0)  # default: 2.0, 0.0, 2.0
        )
        self.airsim_client.setVelocityControllerGains(velocity_gains)

        print("start_state = ",start_state)
        print("trajectory start point: ",self.traj.pos[0,:])
        print("controller ready!")

    def repeat_timer_replanning_callback(self, task, period):
        while self.is_replanning_thread_active:
            task()
            time.sleep(period)

    def replanning_callback(self):
        self.direct_velocity_command()
        self.step += 1
        print("replanning_callback")
        pass


    def start_replanning_callback_thread(self):
        if not self.is_replanning_thread_active:
            self.is_replanning_thread_active = True
            self.replanning_callback_thread.start()
            print("Started replanning callback thread")


    def stop_replanning_callback_thread(self):
        if self.is_replanning_thread_active:
            self.is_replanning_thread_active = False
            self.replanning_callback_thread.join()
            print("Stopped replanning callback thread.")


    def compute_global_optimal_trajectory(self):
        start_state = self.airsim_client.getMultirotorState()
        self.traj = self.global_traj_optimizer.compute_global_optimal_trajectory(start_state)
        return


    def truncate_traj(self,pos,traj,truncate_distance=0.05):
        p1 = traj[self.step,:]
        for k in range(self.step,traj.shape[0]-self.traj_params.horizon):
            p2 = traj[k+1,:]
            dp = (p1-p2) / np.linalg.norm(p1-p2)
            if dp.dot(pos-p2) > truncate_distance:
                self.step = k
                return self.step
        return self.step

    def direct_velocity_command(self):
        idx = np.min([self.step,len(self.traj.t_vec)-1])
        v = self.traj.vel[idx,:]
        duration = self.traj_params.dt_planner
        self.airsim_client.moveByVelocityAsync(v[0],v[1],v[2],duration)


    def update_and_plan(self):
        # retrieve the current state from AirSim
        print("CONTROLLER iteration ",self.planning_cycles,": update_and_plan()")
        self.planning_cycles += 1

        trajectory = self.traj.pos

        pos = self.airsim_client.simGetObjectPose(self.drone_name).position.to_numpy_array()
        k_truncate = self.truncate_traj(pos,trajectory)
        self.step = k_truncate
        print("STEP: ", self.step, "pos: ", pos, "traj[step,:]: ", trajectory[self.step,:])

        # k_truncate == args.n means that the whole trajectory is behind us, and we only issue the last point
        if k_truncate == self.traj_params.n:
            k_truncate = self.traj_params.n - 1
            print('DEBUG: truncating entire trajectory, k_truncate = {}'.format(k_truncate))

        # finally issue the command to AirSim.
        clipped_traj = trajectory[self.step:np.min([self.step+self.traj_params.horizon,len(self.traj.t_vec)]), :]
        if not self.use_vel_constraints:
            # this returns a future, that we do not call .join() on, as we want to re-issue a new command
            # once we compute the next iteration of our high-level planner
            self.airsim_client.moveOnSplineAsync(
                to_airsim_vectors(clipped_traj),
                add_position_constraint = False,
                add_velocity_constraint = True,
                vel_max = self.traj_params.v_max,
                acc_max = self.traj_params.a_max,
                viz_traj = self.viz_traj,
                vehicle_name = self.drone_name,
                replan_from_lookahead = self.traj_params.replan_from_lookahead,
                replan_lookahead_sec = self.traj_params.lookahead_sec)
        else:
            # Compute the velocity as the difference between waypoints
            vel_constraints = self.traj.vel[self.step:np.min([self.step+self.traj_params.horizon,len(self.traj.t_vec)]), :]
            # vel_constraints[1:, :] = trajectory[k_truncate + 1:, :] - trajectory[k_truncate:-1, :]
            # If we use the whole trajectory, the velocity constraint at the first point
            # is computed using the current position
            # if k_truncate == 0:
            #     vel_constraints[0, :] = trajectory[k_truncate, :] - new_state_i
            # else:
            #     vel_constraints[0, :] = trajectory[k_truncate, :] - trajectory[k_truncate - 1, :]

            self.airsim_client.moveOnSplineVelConstraintsAsync(
                to_airsim_vectors(clipped_traj),
                to_airsim_vectors(vel_constraints),
                add_position_constraint=True,
                add_velocity_constraint=True,
                vel_max=self.traj_params.v_max + 10,
                acc_max=self.traj_params.a_max + 10,
                viz_traj=self.viz_traj,
                vehicle_name=self.drone_name
                )
        return


    def run(self):
        # self.start_replanning_callback_thread()
        while self.airsim_client.isApiControlEnabled(vehicle_name=self.drone_name):
            # self.update_and_plan()
            # self.direct_velocity_command()
            self.replanning_callback()
            time.sleep(self.traj_params.dt_planner)
            self.step += 1
            if self.step > len(self.traj.t_vec):
                break
        # self.stop_replanning_callback_thread()

def main(args):
    drone_names = ["drone_1", "drone_2"]
    drone_params = [
        {"r_safe": 0.5,
         "r_coll": 0.5,
         "v_max": 80.0,
         "a_max": 40.0},
        {"r_safe": 0.4,
         "r_coll": 0.3,
         "v_max": 80.0,
         "a_max": 40.0}]

    # set good map-specific conditions
    # if (args.level_name == "Soccer_Field_Easy"):
    #     pass
    # elif (args.level_name == "Soccer_Field_Medium"):
    #     drone_params[0]["v_max"] = 60.0
    #     drone_params[0]["a_max"] = 35.0
    # elif (args.level_name == "ZhangJiaJie_Medium"):
    #     drone_params[0]["v_max"] = 60.0
    #     drone_params[0]["a_max"] = 35.0
    # elif (args.level_name == "Building99_Hard"):
    #     drone_params[0]["v_max"] = 10.0
    #     drone_params[0]["a_max"] = 30.0
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3

    # dr.warmup()
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    racer = HumDrumRacer(
        traj_params=args,
        drone_names=drone_names,
        drone_i=0,  # index of the first drone
        drone_params=drone_params,
        use_vel_constraints=args.vel_constraints)
    # racer.reset_race()

    # racer.load_level(args.level_name)
    # racer.get_ground_truth_gate_poses()
    # print("gate poses",racer.gate_poses_ground_truth)
    # racer.compute_global_optimal_trajectory()

    racer.start_race(args.race_tier)
    racer.initialize_drone()
    racer.takeoff_with_moveOnSpline()

    # racer.start_image_callback_thread()
    racer.start_odometry_callback_thread()
    # racer.start_replanning_callback_thread()

    time.sleep(0.0)  # give opponent a little advantage
    racer.run()

    # racer.stop_image_callback_thread()
    racer.stop_odometry_callback_thread()
    # racer.stop_replanning_callback_thread()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--dt_planner', type=float, default=0.05)
    parser.add_argument('--dt_min', type=float, default=0.05)
    parser.add_argument('--r_safe', type=float, default=0.0)
    parser.add_argument('--v_max', type=float, default=80.0)
    parser.add_argument('--a_max', type=float, default=40.0)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--load_traj_from_file', dest='load_traj_from_file', action='store_true', default=False)
    parser.add_argument('--no_resample', dest='resample', action='store_false', default=True)
    parser.add_argument('--replan_from_lookahead', dest='replan_from_lookahead', action='store_true', default=False)
    parser.add_argument('--lookahead_sec',type=float, default=0.0)
    parser.add_argument('--vel_constraints', dest='vel_constraints', action='store_true', default=False)
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard",
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3"], default="Qualifier_Tier_1")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    main(args)
