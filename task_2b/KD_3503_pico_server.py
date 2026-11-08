#!/usr/bin/env python3
'''
# Team ID:          eYRC#3503
# Theme:            Krishi Drone (KD)
# Author List:      Harsh Patnaik
# Filename:         pico_server.py
# Functions:        __init__ (WayPointServer), disarm (WayPointServer), 
#                   arm (WayPointServer), whycon_callback (WayPointServer), 
#                   odometry_callback (WayPointServer), pid (WayPointServer), 
#                   goal_callback (WayPointServer), cancel_callback (WayPointServer), 
#                   execute_goal_callback (WayPointServer), main
# Global variables: None
'''

import time
import math
from tf_transformations import euler_from_quaternion
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Import custom action
from waypoint_navigation.action import NavToWaypoint

# Import message types
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry


class WayPointServer(Node):
    '''
    Action Server controlling Swift Pico. Receives waypoint goals,
    runs Task 1C PID (x,y,z), monitors stabilization,
    and sends feedback/result, with intermittent yaw correction.
    
    (MODIFIED for Task 2B)
    Now conditionally stabilizes:
        - 3 seconds for pesticide & plant waypoints
        - 0 seconds (visit-only) for all other waypoints
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the WayPointServer action server node. Sets up PID
        constants, state variables, callback groups, publishers,
        subscribers, and the main action server. Arms the drone.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        waypoint_server = WayPointServer()
        '''
        super().__init__('waypoint_server')

        # --- Callback Groups ---
        # self.pid_callback_group: Group for the high-frequency PID timer
        self.pid_callback_group = ReentrantCallbackGroup()
        # self.action_callback_group: Group for the action server callbacks
        self.action_callback_group = ReentrantCallbackGroup()
        # self.odometry_callback_group: Group for odometry subscriber
        self.odometry_callback_group = ReentrantCallbackGroup()
        # self.whycon_callback_group: Group for whycon subscriber
        self.whycon_callback_group = ReentrantCallbackGroup()

        # --- Per-goal state ---
        # self.goal_counter: Tracks the number of goals processed
        self.goal_counter = 0 
        # self.current_goal_abs: Stores the [x,y,z] of the active goal
        self.current_goal_abs = [0.0, 0.0, 0.0]
        # self.point_in_sphere_start_time: Timestamp when drone entered target sphere
        self.point_in_sphere_start_time = None
        # self.time_inside_sphere: Current stabilization streak time
        self.time_inside_sphere = 0.0
        # self.max_time_inside_sphere: Max stabilization time for the current goal
        self.max_time_inside_sphere = 0.0
        # self.last_compact_log_ts: Timestamp for throttling log messages
        self.last_compact_log_ts = 0.0

        # --- Drone state ---
        # self.current_pose: Stores the latest PoseStamped message from WhyCon
        self.current_pose = PoseStamped()
        # self.drone_position: Filtered [x,y,z] absolute position
        self.drone_position = [0.0, 0.0, 0.0]
        # self.current_pos_x,y,z: Individual filtered position components
        self.current_pos_x = 0.0
        self.current_pos_y = 0.0
        self.current_pos_z = 0.0
        # self.base_height: Calibrated Z-level of the ground
        self.base_height = None
        # self.yaw: Current yaw of the drone in degrees
        self.yaw = 0.0
        
        # self.cmd: SwiftMsgs object for sending RC commands
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000 

        # --- PID config (x, y, z) ---
        # self.target_pos: The [x,y,z_relative] PID setpoint
        self.target_pos = [0.0, 0.0, 0.0]
        # self.is_goal_active: Flag to enable/disable the PID loop
        self.is_goal_active = False
        
        # PID constants (Proportional, Integral, Derivative)
        self.Kp_z, self.Ki_z, self.Kd_z = 7.5, 0.002, 95
        self.Kp_x, self.Ki_x, self.Kd_x = 8.0, 0.001, 60.0
        self.Kp_y, self.Ki_y, self.Kd_y = 8.0, 0.001, 60.0
        
        # PID state variables (integral term, previous error, etc.)
        self.iterm_z, self.prev_error_z = 0.0, 0.0
        self.filtered_derivative_z = 0.0 
        self.iterm_x, self.prev_error_x = 0.0, 0.0
        self.filtered_derivative_x = 0.0
        self.iterm_y, self.prev_error_y = 0.0, 0.0
        self.filtered_derivative_y = 0.0
        self.iterm_limit_x = 100.0
        self.iterm_limit_y = 100.0
        self.iterm_limit_z = 100.0
        
        # --- YAW Deadband Config ---
        self.yaw_target_degrees = 0.0
        self.initial_yaw_set = False
        self.yaw_deadband_degrees = 1.0 
        self.yaw_correction_gain = 2.0  
        
        # --- Control Config ---
        self.sample_time = 0.02 # 50Hz PID loop
        self.alpha = 0.8 # EMA filter gain for position
        self.d_filter_alpha = 0.85 # EMA filter gain for D-term
        self.deadband = 0.0
        self.K_ff = 0.01 # Feedforward gain
        
        # --- MODIFICATION: List of "Important" Waypoints ---
        # self.plant_locations: Dictionary of all plant coordinates
        self.plant_locations = {
            'P1A': {'hover': [4.39, 8.98, 29.29], 'spray': [4.5, 8.88, 30.59]},
            'P1B': {'hover': [6.45, 8.79, 28.67], 'spray': [6.85, 8.89, 30.62]},
            'P1C': {'hover': [8.62, 8.80, 28.69], 'spray': [9.13, 8.80, 30.40]},
            'P1D': {'hover': [4.37, 4.57, 29.19], 'spray': [4.53, 4.57, 30.79]},
            'P1E': {'hover': [6.76, 4.49, 28.77], 'spray': [6.45, 4.57, 30.79]},
            'P1F': {'hover': [8.7, 4.53, 28.97],  'spray': [9.1, 4.53, 28.97]}, 
            'P2A': {'hover': [4.43, -4.30, 29.74], 'spray': [4.43, -4.30, 31.1]},
            'P2B': {'hover': [6.68, -4.30, 29.74], 'spray': [7.05, -4.30, 31.52]},
            'P2C': {'hover': [8.93, -4.30, 29.74], 'spray': [9.32, -4.30, 31.06]},
            'P2D': {'hover': [4.58, -8.78, 30.62], 'spray': [4.53, -8.59, 30.76]},
            'P2E': {'hover': [6.76, -8.61, 30.03], 'spray': [7.05, -8.81, 31.52]},
            'P2F': {'hover': [9.17, -8.61, 30.03], 'spray': [9.08, -8.92, 31.93]}
        }
        
        # self.stabilization_points: List of [x,y,z] points requiring 3s dwell
        self.stabilization_points = []
        
        # Add all plant hover and spray points
        for plant_data in self.plant_locations.values():
            self.stabilization_points.append(plant_data['hover'])
            self.stabilization_points.append(plant_data['spray'])

        # Add Pesticide Station 1 points
        self.stabilization_points.append([-9.11, 9.27, 31.27])
        self.stabilization_points.append([-9.07, 9.23, 32.57])
        
        # Add Ground Station for start and final landing
        self.stabilization_points.append([-6.9, 0.09, 32.16])
        
        self.get_logger().info(
            f"--- Server initialized. Will stabilize at {len(self.stabilization_points)} specific waypoints. ---"
        )
        # --- END MODIFICATION ---

        # --- Pub/Sub ---
        # self.command_pub: Publisher for RC commands
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        # self.whycon_sub: Subscriber for WhyCon poses
        self.whycon_sub = self.create_subscription(
            PoseArray, '/whycon/poses', self.whycon_callback, 1,
            callback_group=self.whycon_callback_group
        )
        # self.odom_sub: Subscriber for odometry (to get yaw)
        self.odom_sub = self.create_subscription(
            Odometry, '/rotors/odometry', self.odometry_callback, 10,
            callback_group=self.odometry_callback_group
        )

        # --- Action server ---
        # self._action_server: The main action server
        self._action_server = ActionServer(
            self, NavToWaypoint, 'waypoint_navigation',
            execute_callback=self.execute_goal_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group
        )

        # Arm and start PID timer
        self.armed = False
        self.arm()
        time.sleep(1)
        # self.pid_timer: Timer to run the pid() function at 50Hz
        self.pid_timer = self.create_timer(
            self.sample_time, self.pid, callback_group=self.pid_callback_group
        )
        self.get_logger().info(
            f"--- Waypoint Action Server Initialized (PID @ {1/self.sample_time:.0f}Hz) ---"
        )

    # ---------------- Disarm / Arm ----------------
    def disarm(self):
        '''
        Purpose:
        ---
        Sends the disarm command sequence to the drone.
        Sets throttle to 1000 and AUX4 (arm switch) to 1000.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.disarm()
        '''
        self.get_logger().info("Disarming Drone...")
        self.cmd.rc_aux4 = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_yaw = 1500
        self.command_pub.publish(self.cmd)
        self.armed = False
        self.is_goal_active = False

    def arm(self):
        '''
        Purpose:
        ---
        Sends the arm command sequence to the drone.
        Disarms first, then sets AUX4 (arm switch) to 2000.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.arm()
        '''
        self.get_logger().info("Attempting to Arm Drone...")
        self.disarm()
        time.sleep(0.5)
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1000
        self.command_pub.publish(self.cmd)
        time.sleep(0.1)
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.armed = True
        self.get_logger().info("--- Drone Armed sequence sent ---")

    # ---------------- Callbacks ----------------
    def whycon_callback(self, msg: PoseArray):
        '''
        Purpose:
        ---
        Callback for the /whycon/poses subscriber. It updates the
        drone's current [x, y, z] position using an EMA filter.
        It also calibrates `self.base_height` on the first valid pose.

        Input Arguments:
        ---
        `msg` :  [ geometry_msgs.msg.PoseArray ]
            The pose message from the WhyCon topic.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by the /whycon/poses subscriber)
        '''
        if not msg.poses:
            return 
        
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.poses[0]
        pose = self.current_pose.pose.position

        self.dtime = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if self.current_pose.header.stamp.sec == 0:
            self.get_logger().warn("Waiting for valid WhyCon timestamp...", throttle_duration_sec=1)
            return

        # --- This is the calibration logic ---
        if self.base_height is None and pose.z > 10.0:
            self.base_height = pose.z
            self.get_logger().info(f"--- Calibrated ground level at z = {self.base_height:.2f} ---")
            self.current_pos_z = self.base_height
        # ---

        if self.base_height is not None:
            # Apply Exponential Moving Average (EMA) filter to smooth position
            self.current_pos_z = self.alpha * self.current_pos_z + (1 - self.alpha) * pose.z
            self.current_pos_x = self.alpha * self.current_pos_x + (1 - self.alpha) * pose.x
            self.current_pos_y = self.alpha * self.current_pos_y + (1 - self.alpha) * pose.y
        else:
            self.get_logger().warn("Waiting for valid WhyCon pose to calibrate base_height...",
                                   throttle_duration_sec=1)
            return

        self.drone_position = [self.current_pos_x, self.current_pos_y, self.current_pos_z]

    def odometry_callback(self, msg: Odometry):
        '''
        Purpose:
        ---
        Callback for the /rotors/odometry subscriber. It extracts the
        drone's current yaw from the quaternion orientation.

        Input Arguments:
        ---
        `msg` :  [ nav_msgs.msg.Odometry ]
            The odometry message from the Gazebo simulation.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by the /rotors/odometry subscriber)
        '''
        orientation_q = msg.pose.pose.orientation
        _, _, yaw_rad = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.yaw = math.degrees(yaw_rad)

    # ---------------- PID Loop (with Yaw) ----------------
    def pid(self):
        '''
        Purpose:
        ---
        The main PID control loop, run by `self.pid_timer` (e.g., at 50Hz).
        Calculates PID outputs for X, Y, and Z axes, applies yaw correction,
        and publishes the final RC commands to /drone_command.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by self.pid_timer)
        '''
        # Safety guard: Don't run PID if disarmed, uncalibrated, or no goal
        if not self.armed or self.base_height is None or not self.is_goal_active:
            if not self.armed:
                self.get_logger().warn("PID_GUARD: Drone is DISARMED. Sending low throttle.",
                                       throttle_duration_sec=1)
            elif self.base_height is None:
                self.get_logger().warn("PID_GUARD: Base height is NOT CALIBRATED. Sending low throttle.",
                                       throttle_duration_sec=1)
            elif not self.is_goal_active:
                self.get_logger().info("PID_GUARD: No goal is active. Sending low throttle.",
                                       throttle_duration_sec=1)
            
            if self.armed:
                self.cmd.rc_roll = 1500
                self.cmd.rc_pitch = 1500
                self.cmd.rc_throttle = 1000
                self.cmd.rc_yaw = 1500
                self.command_pub.publish(self.cmd)
            return
            
        target_x, target_y, target_z_relative = self.target_pos

        # --- Z-AXIS (THROTTLE) CALCULATION ---
        relative_position_z = self.current_pos_z - self.base_height
        if relative_position_z > 0: # Prevent drone from going below ground
            relative_position_z = 0.0
        error_z = target_z_relative - relative_position_z
        if abs(error_z) < self.deadband: error_z = 0.0
        
        p_term_z = self.Kp_z * error_z
        self.iterm_z += error_z * self.Ki_z * self.sample_time
        self.iterm_z = max(-self.iterm_limit_z, min(self.iterm_limit_z, self.iterm_z))
        raw_derivative_z = (error_z - self.prev_error_z) / self.sample_time
        # Apply EMA filter to derivative term to reduce noise
        self.filtered_derivative_z = self.d_filter_alpha * self.filtered_derivative_z + \
                                     (1 - self.d_filter_alpha) * raw_derivative_z
        d_term_z = self.Kd_z * self.filtered_derivative_z 
        pid_output_z = p_term_z + self.iterm_z + d_term_z
        # Base throttle is 1530 (hover), output is subtracted
        throttle_pid_output = int(1530 - pid_output_z)
        self.prev_error_z = error_z
        
        # --- X-AXIS CALCULATION (Controls PITCH) ---
        error_x = target_x - self.current_pos_x
        if abs(error_x) < self.deadband: error_x = 0.0
        
        p_term_x = self.Kp_x * error_x
        self.iterm_x += error_x * self.Ki_x * self.sample_time
        self.iterm_x = max(-self.iterm_limit_x, min(self.iterm_limit_x, self.iterm_x))
        raw_derivative_x = (error_x - self.prev_error_x) / self.sample_time
        self.filtered_derivative_x = self.d_filter_alpha * self.filtered_derivative_x + \
                                     (1 - self.d_filter_alpha) * raw_derivative_x
        d_term_x = self.Kd_x * self.filtered_derivative_x 
        pid_output_x = p_term_x + self.iterm_x + d_term_x
        pitch_value = int(1500 + pid_output_x) 
        self.cmd.rc_pitch = max(1000, min(2000, pitch_value))
        self.prev_error_x = error_x

        # --- Y-AXIS CALCULATION (Controls ROLL) ---
        error_y = target_y - self.current_pos_y
        if abs(error_y) < self.deadband: error_y = 0.0
        
        p_term_y = self.Kp_y * error_y
        self.iterm_y += error_y * self.Ki_y * self.sample_time
        self.iterm_y = max(-self.iterm_limit_y, min(self.iterm_limit_y, self.iterm_y))
        raw_derivative_y = (error_y - self.prev_error_y) / self.sample_time
        self.filtered_derivative_y = self.d_filter_alpha * self.filtered_derivative_y + \
                                     (1 - self.d_filter_alpha) * raw_derivative_y
        d_term_y = self.Kd_y * self.filtered_derivative_y 
        pid_output_y = p_term_y + self.iterm_y + d_term_y
        roll_value = int(1500 + pid_output_y)
        self.cmd.rc_roll = max(1000, min(2000, roll_value))
        self.prev_error_y = error_y

        # --- FINAL THROTTLE (with Feedforward) ---
        roll_effort = abs(self.cmd.rc_roll - 1500)
        pitch_effort = abs(self.cmd.rc_pitch - 1500)
        throttle_feedforward = self.K_ff * (roll_effort + pitch_effort)
        throttle_value = int(throttle_pid_output + throttle_feedforward)
        self.cmd.rc_throttle = max(1000, min(2000, throttle_value))

        # --- YAW CORRECTION ---
        error_yaw = self.yaw_target_degrees - self.yaw
        # Handle 360-degree wrap-around
        if error_yaw > 180.0: error_yaw -= 360.0
        elif error_yaw < -180.0: error_yaw += 360.0
        
        if abs(error_yaw) > self.yaw_deadband_degrees:
            correction_output = self.yaw_correction_gain * error_yaw
            self.cmd.rc_yaw = int(1500 - correction_output)
            self.cmd.rc_yaw = max(1000, min(2000, self.cmd.rc_yaw))
        else:
            self.cmd.rc_yaw = 1500
        
        # Publish all commands
        self.command_pub.publish(self.cmd)

        # --- Stabilization Check & Logging ---
        if self.is_goal_active and self.base_height is not None:
            gx, gy, gz = self.current_goal_abs
            cx, cy, cz = self.current_pos_x, self.current_pos_y, self.current_pos_z
            ex, ey, ez = (gx - cx), (gy - cy), (gz - cz)
            in_sphere = (abs(ex) < 0.4 and abs(ey) < 0.4 and abs(ez) < 0.4)

            now_ns = self.get_clock().now().nanoseconds
            if in_sphere:
                # Start timer on first entry into sphere
                if self.point_in_sphere_start_time is None:
                    self.point_in_sphere_start_time = now_ns
                self.time_inside_sphere = (now_ns - self.point_in_sphere_start_time) / 1e9
                if self.time_inside_sphere > self.max_time_inside_sphere:
                    self.max_time_inside_sphere = self.time_inside_sphere
            else:
                # Reset timer if drone leaves sphere
                self.point_in_sphere_start_time = None
                self.time_inside_sphere = 0.0

    # ---------------- Action handlers ----------------
    def goal_callback(self, goal_request):
        '''
        Purpose:
        ---
        Callback that runs when a new goal is *received* by the server.
        It decides whether to accept or reject the goal.

        Input Arguments:
        ---
        `goal_request` :  [ NavToWaypoint.Goal ]
            The goal message sent by the action client.

        Returns:
        ---
        `response` :  [ GoalResponse ]
            Returns GoalResponse.ACCEPT to accept the goal.

        Example call:
        ---
        (Called automatically by the action server)
        '''
        self.get_logger().info('--- Received new goal request. Accepting. ---')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        '''
        Purpose:
        ---
        Callback that runs when a goal *cancellation* is requested.

        Input Arguments:
        ---
        `goal_handle` :  [ rclpy.action.server.ServerGoalHandle ]
            The handle for the goal that is being canceled.

        Returns:
        ---
        `response` :  [ CancelResponse ]
            Returns CancelResponse.ACCEPT to accept the cancellation.

        Example call:
        ---
        (Called automatically by the action server)
        '''
        self.get_logger().info('Received cancel request. Accepting.')
        self.is_goal_active = False
        return CancelResponse.ACCEPT

    def execute_goal_callback(self, goal_handle):
        '''
        Purpose:
        ---
        The main execution callback that runs after a goal is accepted.
        It waits for base height calibration, sets the new PID target,
        and then loops until the stabilization criteria are met.

        Input Arguments:
        ---
        `goal_handle` :  [ rclpy.action.server.ServerGoalHandle ]
            The handle for the currently executing goal.

        Returns:
        ---
        `result` :  [ NavToWaypoint.Result ]
            The final result of the action (e.g., time_taken).

        Example call:
        ---
        (Called automatically by the action server)
        '''
        
        # --- CAPTURE INITIAL YAW on first goal ---
        if not self.initial_yaw_set:
            self.yaw_target_degrees = self.yaw
            self.initial_yaw_set = True
            self.get_logger().warn(
                f"--- Captured Initial Yaw. Target Set to: {self.yaw_target_degrees:.2f} degrees ---"
            )
        
        # --- FIX: WAIT FOR BASE HEIGHT CALIBRATION ---
        self.get_logger().info("Waiting for base height calibration...")
        wait_rate = self.create_rate(10) # Check 10 times/sec
        
        while self.base_height is None and rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled while waiting for base height.')
                return NavToWaypoint.Result()
                
            self.get_logger().warn(
                "Waiting for valid WhyCon pose to calibrate base_height...", 
                throttle_duration_sec=1.0
            )
            wait_rate.sleep()
        
        self.get_logger().info(f"--- Base height calibrated at {self.base_height:.2f}. Proceeding with goal. ---")
        # --- *** END FIX *** ---

        start_time_ns = self.get_clock().now().nanoseconds
        goal_point = goal_handle.request.waypoint
        goal_x, goal_y, goal_z_abs = goal_point.x, goal_point.y, goal_point.z

        # Per-goal state setup
        self.goal_counter += 1
        self.current_goal_abs = [goal_x, goal_y, goal_z_abs]
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0.0
        self.max_time_inside_sphere = 0.0
        self.last_compact_log_ts = 0.0

        self.get_logger().info(
            f'--- EXECUTING GOAL #{self.goal_counter} to Absolute WP: [{goal_x:.2f}, {goal_y:.2f}, {goal_z_abs:.2f}] ---'
        )
        
        # --- Check if this is a "Stabilization" or "Visit-Only" goal ---
        is_stabilize_goal = False
        for sp in self.stabilization_points:
            dist = math.sqrt((goal_x - sp[0])**2 + (goal_y - sp[1])**2 + (goal_z_abs - sp[2])**2)
            if dist < 0.01: # Check if points are virtually identical
                is_stabilize_goal = True
                break
        
        if is_stabilize_goal:
            self.get_logger().warn(f"*** This is a STABILIZATION goal. Requiring 3s dwell. ***")
            required_dwell_time = 3.0
        else:
            self.get_logger().info(f"--- This is a VISIT-ONLY goal. Requiring 0s dwell. ---")
            required_dwell_time = 0.0
        # --- END MODIFICATION ---

        # Convert absolute Z goal to relative Z for the PID
        relative_goal_height = goal_z_abs - self.base_height
        target_z_relative = relative_goal_height 

        # Reset PID states for the new goal
        self.get_logger().info("Resetting X, Y, Z PID states for new goal.")
        self.iterm_x, self.prev_error_x, self.filtered_derivative_x = 0.0, 0.0, 0.0
        self.iterm_y, self.prev_error_y, self.filtered_derivative_y = 0.0, 0.0, 0.0
        self.iterm_z, self.prev_error_z, self.filtered_derivative_z = 0.0, 0.0, 0.0
        
        self.target_pos = [goal_x, goal_y, target_z_relative]
        self.get_logger().warn(
            f"--- SETPOINT: Abs WP [{goal_x:.2f}, {goal_y:.2f}, {goal_z_abs:.2f}] "
            f"| Relative Z target for PID: [{target_z_relative:.2f}] ---"
        )

        # Activate the PID loop
        self.get_logger().info("--- ACTIVATING PID CONTROLLER (is_goal_active = True) ---")
        self.is_goal_active = True

        feedback_msg = NavToWaypoint.Feedback()
        stabilized = False
        error_margin = 0.4 # Stabilization sphere radius

        # Main monitoring loop (runs at 20Hz)
        loop_rate = self.create_rate(20)
        while rclpy.ok():
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.is_goal_active = False
                self.get_logger().info('Goal was canceled during execution.')
                return NavToWaypoint.Result()

            # Check for disarm
            if not self.armed:
                self.get_logger().warn("Drone disarmed during goal execution. Aborting.")
                goal_handle.abort()
                self.is_goal_active = False
                return NavToWaypoint.Result()

            # Publish feedback
            try:
                feedback_msg.current_pose.header = self.current_pose.header
                feedback_msg.current_pose.pose = self.current_pose.pose
                # Hack: Use 'sec' field to send stabilization time
                if feedback_msg.current_pose.header.stamp.sec > 0:
                    feedback_msg.current_pose.header.stamp.sec = int(self.max_time_inside_sphere)
                goal_handle.publish_feedback(feedback_msg)
            except Exception as e:
                self.get_logger().error(f"Feedback publish failed: {e}")
                goal_handle.abort()
                self.is_goal_active = False
                return NavToWaypoint.Result()

            # Current ABS positions
            current_x_abs, current_y_abs, current_z_abs = self.drone_position

            # Errors vs ABS target
            err_x = goal_x - current_x_abs
            err_y = goal_y - current_y_abs
            err_z = goal_z_abs - current_z_abs
            err_norm = (err_x**2 + err_y**2 + err_z**2) ** 0.5

            # In-sphere check
            drone_in_sphere = (
                abs(err_x) < error_margin and
                abs(err_y) < error_margin and
                abs(err_z) < error_margin
            )

            # --- New Success Condition ---
            if drone_in_sphere and self.time_inside_sphere >= required_dwell_time:
                self.get_logger().info(
                    f'>>> SENDING STATUS_SUCCEEDED for Goal #{self.goal_counter} '
                    f'(RequiredDwell={required_dwell_time:.1f}s, ActualDwell={self.time_inside_sphere:.2f}s)'
                )
                stabilized = True
                break # Exit the while loop
            # --- END MODIFICATION ---

            # Throttle-logged status message
            self.get_logger().info(
                f"[GOAL {self.goal_counter}] "
                f"Targ=[{goal_x:.2f},{goal_y:.2f},{goal_z_abs:.2f}] "
                f"Err=[{err_x:+.2f},{err_y:+.2f},{err_z:+.2f}] |‖e‖|={err_norm:.2f} "
                f"In±0.4={drone_in_sphere} streak={self.time_inside_sphere:.2f}s "
                f"(Req: {required_dwell_time:.1f}s)",
                throttle_duration_sec=1.0
            )

            # Blocking sleep @ 20 Hz
            try:
                loop_rate.sleep()
            except Exception as e:
                self.get_logger().error(f"Rate sleep failed: {e}")
                goal_handle.abort()
                self.is_goal_active = False
                return NavToWaypoint.Result()

        # --- Loop exited ---
        self.get_logger().info("--- DEACTIVATING PID CONTROLLER (is_goal_active = False) ---")
        self.is_goal_active = False
        end_time_ns = self.get_clock().now().nanoseconds
        time_taken = (end_time_ns - start_time_ns) / 1e9
        result = NavToWaypoint.Result()
        result.time_taken = time_taken

        if stabilized:
            self.get_logger().info(
                f'Goal Succeeded! Total time: {time_taken:.2f}s | '
                f'Max dwell in ±0.4: {self.max_time_inside_sphere:.2f}s'
            )
            goal_handle.succeed()
        else:
            self.get_logger().warn('Goal Aborted (Loop exited unexpectedly).')
            goal_handle.abort()

        return result


def main(args=None):
    '''
    Purpose:
    ---
    Initializes the ROS 2 system and the WayPointServer node.
    It uses a MultiThreadedExecutor to spin the node, allowing it to
    handle multiple callbacks concurrently (e.g., PID timer and action server).

    Input Arguments:
    ---
    `args` :  [ list, optional ]
        Command-line arguments passed to rclpy.init().

    Returns:
    ---
    None

    Example call:
    ---
    (Called automatically when the script is executed)
    '''
    rclpy.init(args=args)
    waypoint_server = None
    try:
        waypoint_server = WayPointServer()
        # Use a MultiThreadedExecutor to handle all the re-entrant callbacks
        executor = MultiThreadedExecutor()
        rclpy.spin(waypoint_server, executor=executor)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if waypoint_server:
            waypoint_server.get_logger().error(f"Unhandled exception: {e}")
        else:
            print(f"Unhandled exception during node initialization: {e}")
    finally:
        if waypoint_server and rclpy.ok():
            waypoint_server.get_logger().info('Shutting down server node.')
            waypoint_server.disarm()
            waypoint_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()