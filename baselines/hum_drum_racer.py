from baseline_racer import BaselineRacer
from gtp_visualize import *
from utils import to_airsim_vector, to_airsim_vectors
import airsimneurips as airsim
import argparse
import gtp
import numpy as np
import time
# Use non interactive matplotlib backend
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #3d plotting

# Julia interface to JuMP / Ipopt
import julia
from julia import Main
from julia import DroneRacing as dr

class GlobalTrajectoryOptimizer():
    def __init__(self,traj_params, drone_params, gate_poses, gate_inner_dims, gate_outer_dims):
        self.traj_params = traj_params
        self.drone_params = drone_params
        self.gate_poses = gate_poses
        self.gate_inner_dims = gate_inner_dims
        self.gate_outer_dims = gate_outer_dims

    def shuffle_gates(self,current_gate_idx):
        # TODO implement array rotation
        pass

    def compute_global_optimal_trajectory(self,start_state,
        **kwargs
    ):
        """
        Compute a globally optimal trajectory
        """
        # Get current state
        start_pos = start_state.kinematics_estimated.position
        start_vel = start_state.kinematics_estimated.linear_velocity
        ####################### Pass information to Julia ######################
        Main.start_pos = [start_pos.x_val, start_pos.y_val, start_pos.z_val]
        Main.start_vel = [start_vel.x_val, start_vel.y_val, start_vel.z_val]
        Main.traj_opt_model = dr.VariableTimeParticleTrajOptModel(
            start_pos = [start_pos.x_val, start_pos.y_val, start_pos.z_val],
            start_vel = [start_vel.x_val, start_vel.y_val, start_vel.z_val],
            # n = 10,
            MAX_VEL = self.drone_params["v_max"],
            MAX_ACCEL = self.drone_params["a_max"],
            # dt_min = 0.025
            **kwargs
        )
        for pose in self.gate_poses:
            pos = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
            orientation = [pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val]
            inner_width = [self.gate_inner_dims.x_val, self.gate_inner_dims.y_val, self.gate_inner_dims.z_val]
            Main.gate = dr.Gate3D(pos,orientation,inner_width)
            Main.eval("push!(traj_opt_model.gates, gate)")
        ########################################################################
        ######################### Optimize Ipopt model #########################
        Main.JuMP_model = dr.formulate_global_traj_opt_problem(Main.traj_opt_model)
        Main.eval("DroneRacing.optimize_trajectory!(traj_opt_model,JuMP_model)")
        x = dr.get_x(Main.traj_opt_model,Main.JuMP_model)
        v = dr.get_v(Main.traj_opt_model,Main.JuMP_model)
        a = dr.get_a(Main.traj_opt_model,Main.JuMP_model)
        dt = dr.get_dt(Main.traj_opt_model,Main.JuMP_model)
        return x,v,a,dt
        ########################################################################

class HumDrumRacer(BaselineRacer):
    def __init__(self, traj_params, drone_names, drone_i, drone_params,
                 use_vel_constraints=False,
                 plot_gtp=False):
        super().__init__(drone_name=drone_names[drone_i],
            viz_traj=True)
        self.drone_names = drone_names
        self.drone_i = drone_i
        self.drone_params = drone_params
        self.traj_params = traj_params
        self.use_vel_constraints = use_vel_constraints
        self.plot_gtp = plot_gtp
        self.controller = None
        self.global_traj_optimizer = GlobalTrajectoryOptimizer(traj_params,drone_params)


        # for plotting: Just some fig, ax and line objects to keep track of
        if self.plot_gtp:
            self.fig, self.ax = plt.subplots()
            self.line_state = None
            self.lines = [None] * 2
            # plot 3d track just once for visualization
            self.fig2 = plt.figure(2)
            self.ax3d = self.fig2.add_subplot(111, projection='3d')

        print("controller ready!")
        if (self.traj_params.blocking):
            print("   with blocking behavior activated")
        # self.image_num = 0

    def compute_global_optimal_trajectory(self):
        """
        Compute a globally optimal trajectory
        """
        # Get current state
        start_state = self.airsim_client.getMultirotorState()
        self.global_traj_optimizer.compute_global_optimal_trajectory(start_state)
        self.controller_log("successfully optimized trajectory!")


    def update_and_plan(self):
        # retrieve the current state from AirSim
        position_airsim = []
        for drone_name in self.drone_names:
            position_airsim.append(self.airsim_client.simGetObjectPose(drone_name).position)

        state = np.array([position.to_numpy_array() for position in position_airsim])

        if self.plot_gtp:
            # plot or update the state
            if self.line_state is None:
                self.line_state, = plot_state(self.ax, state)
            else:
                replot_state(self.line_state, state)

        trajectory = self.controller.iterative_br(self.drone_i, state)

        # now, let's issue the new trajectory to the trajectory planner
        # fetch the current state first, to see, if our trajectory is still planned for ahead of us
        new_state_i = self.airsim_client.simGetObjectPose(self. drone_name).position.to_numpy_array()

        if self.plot_gtp:
            replot_state(self.line_state, state)

        # as we move while computing the trajectory,
        # make sure that we only issue the part of the trajectory, that is still ahead of us
        k_truncate, _ = self.controller.truncate(new_state_i, trajectory[:, :])
        # print("k_truncate: ", k_truncate)

        # k_truncate == args.n means that the whole trajectory is behind us, and we only issue the last point
        if k_truncate == self.traj_params.n:
            k_truncate = self.traj_params.n - 1
            print('DEBUG: truncating entire trajectory, k_truncate = {}'.format(k_truncate))

        if self.plot_gtp:
            # For our 2D trajectory, let's plot or update
            if self.lines[self.drone_i] is None:
                self.lines[self.drone_i], = plot_trajectory_2d(self.ax, trajectory[k_truncate:, :])
            else:
                replot_trajectory_2d(self.lines[self.drone_i], trajectory[k_truncate:, :])

        # finally issue the command to AirSim.
        if not self.use_vel_constraints:
            # this returns a future, that we do not call .join() on, as we want to re-issue a new command
            # once we compute the next iteration of our high-level planner

            self.airsim_client.moveOnSplineAsync(
                to_airsim_vectors(trajectory[k_truncate:, :]),
                add_position_constraint=False,
                add_velocity_constraint=True,
                vel_max=self.drone_params[self.drone_i]["v_max"],
                acc_max=self.drone_params[self.drone_i]["a_max"],
                viz_traj=self.viz_traj,
                vehicle_name=self.drone_name,
                replan_from_lookahead=False,
                replan_lookahead_sec=0.0)
        else:
            # Compute the velocity as the difference between waypoints
            vel_constraints = np.zeros_like(trajectory[k_truncate:, :])
            vel_constraints[1:, :] = trajectory[k_truncate + 1:, :] - trajectory[k_truncate:-1, :]
            # If we use the whole trajectory, the velocity constraint at the first point
            # is computed using the current position
            if k_truncate == 0:
                vel_constraints[0, :] = trajectory[k_truncate, :] - new_state_i
            else:
                vel_constraints[0, :] = trajectory[k_truncate, :] - trajectory[k_truncate - 1, :]

            self.airsim_client.moveOnSplineVelConstraintsAsync(
                to_airsim_vectors(trajectory[k_truncate:, :]),
                to_airsim_vectors(vel_constraints),
                add_position_constraint=True,
                add_velocity_constraint=True,
                vel_max=self.drone_params[self.drone_i]["v_max"],
                acc_max=self.drone_params[self.drone_i]["a_max"],
                viz_traj=self.viz_traj,
                vehicle_name=self.drone_name)

        if self.plot_gtp:
            # refresh the updated plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # save figure
            # figure_file_name = '/home/ericcristofalo/Documents/game_of_drones/log/image_' + str(self.image_num) + '.png'
            # print('saving image: ', figure_file_name)
            # self.fig.savefig(figure_file_name)
            # self.image_num = self.image_num + 1
            # # set the figure bounds around drone i for visualization
            # self.fig.xlim(state[self.drone_i][1] - 10, state[self.drone_i][1] + 10)
            # self.fig.ylim(state[self.drone_i][0] - 10, state[self.drone_i][0] + 10)

    def run(self):
        self.get_ground_truth_gate_poses()

        # We pretend we have two different controllers for the drones,
        # so let's instantiate  two
        self.controller = gtp.IBRController(self.traj_params, self.drone_params, self.gate_poses_ground_truth)

        if self.plot_gtp:
            # Let's plot the gates, and the fitted track.
            plot_gates_2d(self.ax, self.gate_poses_ground_truth)
            plot_track(self.ax, self.controller.track)
            plot_track_arrows(self.ax, self.controller.track)
            plt.ion()  # turn on interactive mode to plot while running script
            self.fig.show()

            plot_track3d(self.ax3d, self.controller.track)
            self.fig2.show()

        while self.airsim_client.isApiControlEnabled(vehicle_name=self.drone_name):
            self.update_and_plan()

def main(args):
    drone_names = ["drone_1", "drone_2"]
    drone_params = [
        {"r_safe": 0.5,
         "r_coll": 0.5,
         "v_max": 80.0,
         "a_max": 40.0},
        {"r_safe": 0.4,
         "r_coll": 0.3,
         "v_max": 20.0,
         "a_max": 10.0}]

    # set good map-specific conditions
    if (args.level_name == "Soccer_Field_Easy"):
        pass
    elif (args.level_name == "Soccer_Field_Medium"):
        drone_params[0]["v_max"] = 60.0
        drone_params[0]["a_max"] = 35.0
    elif (args.level_name == "ZhangJiaJie_Medium"):
        drone_params[0]["v_max"] = 60.0
        drone_params[0]["a_max"] = 35.0
    elif (args.level_name == "Building99_Hard"):
        drone_params[0]["v_max"] = 10.0
        drone_params[0]["a_max"] = 30.0
    elif (args.level_name == "Qualifier_Tier_1"):
        pass
    elif (args.level_name == "Qualifier_Tier_2"):
        pass
    elif (args.level_name == "Qualifier_Tier_3"):
        pass

    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    racer = HumDrumRacer(
        traj_params=args,
        drone_names=drone_names,
        drone_i=0,  # index of the first drone
        drone_params=drone_params,
        use_vel_constraints=args.vel_constraints,
        plot_gtp=args.plot_gtp)

    racer.load_level(args.level_name)
    racer.get_ground_truth_gate_poses()
    print("gate poses",racer.gate_poses_ground_truth)
    racer.compute_global_optimal_trajectory()

    racer.start_race(args.race_tier)
    racer.initialize_drone()
    racer.takeoff_with_moveOnSpline()
    time.sleep(0.0)  # give opponent a little advantage
    racer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--n', type=int, default=14)
    parser.add_argument('--blocking_behavior', dest='blocking', action='store_true', default=False)
    parser.add_argument('--vel_constraints', dest='vel_constraints', action='store_true', default=False)
    parser.add_argument('--plot_gtp', dest='plot_gtp', action='store_true', default=False)
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard",
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3"], default="ZhangJiaJie_Medium")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    main(args)
