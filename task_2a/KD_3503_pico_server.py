#!/usr/bin/env python3
'''
# Team ID:          eYRC#3503
# Theme:            Krishi Drone (KD)
# Author List:      Harsh Patnaik
# Filename:         KD_3503_pico_server.py
# Functions:        __init__, whycon_callback, arm, disarm, pid, odometry_callback,
#                   goal_callback, cancel_callback, execute_goal_callback, main
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
    runs Task 1C PID (x,y,z), monitors stabilization (±0.4 for 3 s),
    sends feedback/result, with intermittent yaw correction relative to STARTING yaw.
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the WayPointServer node. Sets up callback groups,
        initializes PID constants and state variables, creates publishers,
        subscribers, the action server, and arms the drone.

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

        # Callback groups
        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        self.odometry_callback_group = ReentrantCallbackGroup()
        self.whycon_callback_group = ReentrantCallbackGroup()

        # Per-goal dwell/log state (shared across PID and action loop)
        self.goal_counter = 0 # Counter for tracking the number of goals executed
        
        # self.current_goal_abs: [x,y,z] absolute target for the *current* action goal
        self.current_goal_abs = [0.0, 0.0, 0.0]
        
        # self.point_in_sphere_start_time: ns timestamp when drone *first* entered the stabilization sphere
        self.point_in_sphere_start_time = None
        
        # self.time_inside_sphere: Current continuous time (s) spent inside the stabilization sphere
        self.time_inside_sphere = 0.0
        
        # self.max_time_inside_sphere: Max continuous time (s) spent inside the sphere during a goal
        self.max_time_inside_sphere = 0.0
        
        # self.last_compact_log_ts: Wall-time throttle for compact goal logging (s)
        self.last_compact_log_ts = 0.0

        # Drone state
        self.current_pose = PoseStamped() # PoseStamped for action feedback
        
        # self.drone_position: [x, y, z] filtered absolute position from WhyCon
        self.drone_position = [0.0, 0.0, 0.0]
        self.current_pos_x = 0.0 # Filtered absolute X
        self.current_pos_y = 0.0 # Filtered absolute Y
        self.current_pos_z = 0.0 # Filtered absolute Z
        
        # self.base_height: Calibrated Z-level (absolute) of the ground, detected on startup
        self.base_height = None
        
        # self.yaw: Current yaw in degrees, continuously updated by odometry_callback
        self.yaw = 0.0
        self.dtime = 0.0 # Timestamp from WhyCon message

        # RC command
        # self.cmd: SwiftMsgs object holding the current RC commands (1000-2000)
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000  # start disarmed

        # --- PID config (x, y, z) - IMPORTED FROM pico_controller_PID.py ---
        # self.target_pos: [x_abs, y_abs, z_rel] target setpoint for the PID controller
        self.target_pos = [0.0, 0.0, 0.0]
        
        # self.is_goal_active: Flag to enable/disable PID calculations in self.pid()
        self.is_goal_active = False
        
        # Z-Axis / Throttle (Values from pico_controller_PID.py)
        self.Kp_z, self.Ki_z, self.Kd_z = 7.5, 0.002, 95
        # X-Axis / Pitch (Values from pico_controller_PID.py)
        self.Kp_x, self.Ki_x, self.Kd_x = 8.0, 0.001, 60.0
        # Y-Axis / Roll (Values from pico_controller_PID.py)
        self.Kp_y, self.Ki_y, self.Kd_y = 8.0, 0.001, 60.0

        # PID State Variables (Z-Axis)
        # self.iterm_z, self.prev_error_z: State variables for the Z-axis PID controller
        self.iterm_z, self.prev_error_z = 0.0, 0.0
        self.filtered_derivative_z = 0.0 
        # PID State Variables (X-Axis)
        self.iterm_x, self.prev_error_x = 0.0, 0.0
        self.filtered_derivative_x = 0.0
        # PID State Variables (Y-Axis)
        self.iterm_y, self.prev_error_y = 0.0, 0.0
        self.filtered_derivative_y = 0.0

        # I-Term Limits (Values from pico_controller_PID.py)
        self.iterm_limit_x = 100.0
        self.iterm_limit_y = 100.0
        self.iterm_limit_z = 100.0
        # --- END PID CONFIG ---
        
        # --- YAW Deadband Config (From original server) ---
        # self.yaw_target_degrees: The yaw angle (degrees) captured when the *first* goal was received
        self.yaw_target_degrees = 0.0
        
        # self.initial_yaw_set: Flag to ensure the initial yaw is captured only once
        self.initial_yaw_set = False
        self.yaw_deadband_degrees = 1.0 # No correction inside +/- 1 degrees
        self.yaw_correction_gain = 2.0  # Gentle proportional gain
        # --- END YAW ---

        # --- Control Config (From pico_controller_PID.py) ---
        self.sample_time = 0.02        # PID loop period
        self.alpha = 0.8               # smoothing factor for position feedback
        self.d_filter_alpha = 0.85     # D-Term filter strength
        self.deadband = 0.0            # tolerance to ignore small oscillations
        self.K_ff = 0.01             # Feedforward gain for throttle
        # --- END Control Config ---

        # Pub/Sub
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.whycon_sub = self.create_subscription(
            PoseArray, '/whycon/poses', self.whycon_callback, 1,
            callback_group=self.whycon_callback_group
        )
        
        # YAW CONTROL ENABLED
        self.odom_sub = self.create_subscription(
            Odometry, '/rotors/odometry', self.odometry_callback, 10,
            callback_group=self.odometry_callback_group
        )

        # Action server
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
        Sends the disarm command (RC_AUX4 = 1000) and sets throttle to 1000.
        Updates the internal armed state.

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
        self.cmd.rc_yaw = 1500 # Reset yaw on disarm
        self.command_pub.publish(self.cmd)
        self.armed = False
        self.is_goal_active = False

    def arm(self):
        '''
        Purpose:
        ---
        Executes the arming sequence for the drone.
        Sends Disarm -> Wait -> Arm (RC_AUX4 = 2000).

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
        self.disarm() # Ensure drone is disarmed first
        time.sleep(0.5)
        # Set neutral commands
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1000
        self.command_pub.publish(self.cmd)
        time.sleep(0.1)
        # Send arm command
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.armed = True
        self.get_logger().info("--- Drone Armed sequence sent ---")

    # ---------------- Callbacks ----------------
    def whycon_callback(self, msg: PoseArray):
        '''
        Purpose:
        ---
        Processes incoming WhyCon PoseArray messages. It performs one-time
        calibration of `base_height` (ground Z-level) and continuously
        updates the drone's filtered absolute position (`current_pos_x`,
        `current_pos_y`, `current_pos_z`).

        Input Arguments:
        ---
        `msg` : [ geometry_msgs.msg.PoseArray ]
            The PoseArray message from the '/whycon/poses' topic.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by ROS 2 Subscription)
        '''
        if not msg.poses:
            return # Ignore empty messages
        
        # Update current pose for feedback
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.poses[0]
        pose = self.current_pose.pose.position

        self.dtime = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if self.current_pose.header.stamp.sec == 0:
            self.get_logger().warn("Waiting for valid WhyCon timestamp...", throttle_duration_sec=1)
            return

        # Calibrate base_height on the first valid pose
        if self.base_height is None and pose.z > 10.0:
            self.base_height = pose.z
            self.get_logger().info(f"--- Calibrated ground level at z = {self.base_height:.2f} ---")
            self.current_pos_z = self.base_height # Initialize filtered z

        # Apply low-pass filtering only after calibration
        if self.base_height is not None:
            self.current_pos_z = self.alpha * self.current_pos_z + (1 - self.alpha) * pose.z
            self.current_pos_x = self.alpha * self.current_pos_x + (1 - self.alpha) * pose.x
            self.current_pos_y = self.alpha * self.current_pos_y + (1 - self.alpha) * pose.y
        else:
            self.get_logger().warn("Waiting for valid WhyCon pose to calibrate base_height...",
                                   throttle_duration_sec=1)
            return

        # Update the composite position list
        self.drone_position = [self.current_pos_x, self.current_pos_y, self.current_pos_z]

    def odometry_callback(self, msg: Odometry):
        '''
        Purpose:
        ---
        Processes incoming Odometry messages *only* to extract and update
        the drone's current yaw angle (`self.yaw`) in degrees.

        Input Arguments:
        ---
        `msg` : [ nav_msgs.msg.Odometry ]
            The Odometry message from the '/rotors/odometry' topic.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by ROS 2 Subscription)
        '''
        # This callback now *only* updates the current yaw state.
        orientation_q = msg.pose.pose.orientation
        _, _, yaw_rad = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.yaw = math.degrees(yaw_rad) # Update self.yaw in degrees

    # ---------------- PID Loop (with Yaw) ----------------
    def pid(self):
        '''
        Purpose:
        ---
        The main PID control loop, executed on a timer. If a goal is active,
        it calculates the required Roll, Pitch, Throttle, and Yaw commands
        to reach the `self.target_pos` setpoint. It also publishes the
        commands and handles logging.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by ROS 2 Timer)
        '''
        
        # Guard clause: Do not run PID if disarmed, not calibrated, or no goal is active
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
            
            # Send neutral/safe commands if armed but not active
            if self.armed:
                self.cmd.rc_roll = 1500
                self.cmd.rc_pitch = 1500
                self.cmd.rc_throttle = 1000 # Send low throttle
                self.cmd.rc_yaw = 1500 # Keep yaw centered
                self.command_pub.publish(self.cmd)
            return
            
        # Get targets from self.target_pos (set by execute_goal_callback)
        target_x, target_y, target_z_relative = self.target_pos

        # -----------------------------------------------------------------
        # --- Z-AXIS (THROTTLE) CALCULATION (From pico_controller_PID.py) ---
        # -----------------------------------------------------------------
        
        # Calculate current height relative to the calibrated ground
        relative_position_z = self.current_pos_z - self.base_height
        if relative_position_z > 0: # Clamp if drone is "above" ground (e.g., in simulation)
            relative_position_z = 0.0

        error_z = target_z_relative - relative_position_z
        if abs(error_z) < self.deadband:
            error_z = 0.0

        p_term_z = self.Kp_z * error_z
        
        # Calculate I-term with anti-windup
        self.iterm_z += error_z * self.Ki_z * self.sample_time
        self.iterm_z = max(-self.iterm_limit_z, min(self.iterm_limit_z, self.iterm_z))

        # Calculate filtered D-term
        raw_derivative_z = (error_z - self.prev_error_z) / self.sample_time
        self.filtered_derivative_z = self.d_filter_alpha * self.filtered_derivative_z + \
                                     (1 - self.d_filter_alpha) * raw_derivative_z
        d_term_z = self.Kd_z * self.filtered_derivative_z 

        pid_output_z = p_term_z + self.iterm_z + d_term_z

        # Store Z-PID output (pre-feedforward)
        # 1530 is the hover-point, and output is subtracted (inverted)
        throttle_pid_output = int(1530 - pid_output_z)
        self.prev_error_z = error_z
        
        # -----------------------------------------------------------------
        # --- X-AXIS CALCULATION (Controls PITCH) (From pico_controller_PID.py) ---
        # -----------------------------------------------------------------
        error_x = target_x - self.current_pos_x
        if abs(error_x) < self.deadband:
            error_x = 0.0

        p_term_x = self.Kp_x * error_x

        # Calculate I-term with anti-windup
        self.iterm_x += error_x * self.Ki_x * self.sample_time
        self.iterm_x = max(-self.iterm_limit_x, min(self.iterm_limit_x, self.iterm_x))

        # Calculate filtered D-term
        raw_derivative_x = (error_x - self.prev_error_x) / self.sample_time
        self.filtered_derivative_x = self.d_filter_alpha * self.filtered_derivative_x + \
                                     (1 - self.d_filter_alpha) * raw_derivative_x
        d_term_x = self.Kd_x * self.filtered_derivative_x 
        
        pid_output_x = p_term_x + self.iterm_x + d_term_x

        # Apply PID output to neutral 1500
        pitch_value = int(1500 + pid_output_x) 
        self.cmd.rc_pitch = max(1000, min(2000, pitch_value)) # Clamp output
        self.prev_error_x = error_x

        # -----------------------------------------------------------------
        # --- Y-AXIS CALCULATION (Controls ROLL) (From pico_controller_PID.py) ---
        # -----------------------------------------------------------------
        error_y = target_y - self.current_pos_y
        if abs(error_y) < self.deadband:
            error_y = 0.0

        p_term_y = self.Kp_y * error_y

        # Calculate I-term with anti-windup
        self.iterm_y += error_y * self.Ki_y * self.sample_time
        self.iterm_y = max(-self.iterm_limit_y, min(self.iterm_limit_y, self.iterm_y))
        
        # Calculate filtered D-term
        raw_derivative_y = (error_y - self.prev_error_y) / self.sample_time
        self.filtered_derivative_y = self.d_filter_alpha * self.filtered_derivative_y + \
                                     (1 - self.d_filter_alpha) * raw_derivative_y
        d_term_y = self.Kd_y * self.filtered_derivative_y 

        pid_output_y = p_term_y + self.iterm_y + d_term_y

        # Apply PID output to neutral 1500
        roll_value = int(1500 + pid_output_y)
        self.cmd.rc_roll = max(1000, min(2000, roll_value))
        self.prev_error_y = error_y

        # -----------------------------------------------------------------
        # --- FINAL THROTTLE (with Feedforward) (From pico_controller_PID.py) ---
        # -----------------------------------------------------------------
        
        # Get the 'control effort' for roll and pitch (how far from neutral)
        roll_effort = abs(self.cmd.rc_roll - 1500)
        pitch_effort = abs(self.cmd.rc_pitch - 1500)
        
        # Add extra throttle proportional to the *combined* roll/pitch effort
        throttle_feedforward = self.K_ff * (roll_effort + pitch_effort)

        # Add the FF boost to the original PID output
        throttle_value = int(throttle_pid_output + throttle_feedforward)
        
        # Clamp the final value
        self.cmd.rc_throttle = max(1000, min(2000, throttle_value))

        # -----------------------------------------------------------------
        # --- YAW CORRECTION (From original pico_server.py) ---
        # -----------------------------------------------------------------
        
        # self.yaw_target_degrees was set once on the first goal
        error_yaw = self.yaw_target_degrees - self.yaw
        
        # Handle angle wrap-around (e.g., -170 to +170)
        if error_yaw > 180.0:
            error_yaw -= 360.0
        elif error_yaw < -180.0:
            error_yaw += 360.0

        # Apply correction only if error is outside the deadband
        if abs(error_yaw) > self.yaw_deadband_degrees:
            correction_output = self.yaw_correction_gain * error_yaw
            
            # Apply correction: 1500 - output (inverted)
            self.cmd.rc_yaw = int(1500 - correction_output)
            self.cmd.rc_yaw = max(1000, min(2000, self.cmd.rc_yaw))
        else:
            # Inside deadband, just hold 1500
            self.cmd.rc_yaw = 1500
        # --- END YAW ---


        # Publish all commands
        self.command_pub.publish(self.cmd)

        # Logging (from original server, adapted for new variables)
        self.get_logger().info(
            f"[PID_DEBUG] AbsPos: [{self.current_pos_x:6.2f}, {self.current_pos_y:6.2f}, {self.current_pos_z:6.2f}]\n"
            f"[PID_DEBUG] RelPos Z: {relative_position_z:6.2f} | RelTarg Z: {target_z_relative:6.2f}\n"
            f"[PID_DEBUG] Err: [X:{error_x:6.2f} Y:{error_y:6.2f} Z:{error_z:6.2f}] | YAW Angle:{self.yaw:6.2f}, Err:{error_yaw:6.2f} \n"
            f"CMD: [R:{self.cmd.rc_roll:4d}, P:{self.cmd.rc_pitch:4d}, T:{self.cmd.rc_throttle:4d}, Y:{self.cmd.rc_yaw:4d}]",
            throttle_duration_sec=0.2
        )

        # Compact dwell log (monitors stabilization for the current goal)
        if self.is_goal_active and self.base_height is not None:
            gx, gy, gz = self.current_goal_abs
            cx, cy, cz = self.current_pos_x, self.current_pos_y, self.current_pos_z
            ex, ey, ez = (gx - cx), (gy - cy), (gz - cz)
            in_sphere = (abs(ex) < 0.4 and abs(ey) < 0.4 and abs(ez) < 0.4)

            now_ns = self.get_clock().now().nanoseconds
            if in_sphere:
                # If we just entered the sphere, log the start time
                if self.point_in_sphere_start_time is None:
                    self.point_in_sphere_start_time = now_ns
                
                # Calculate continuous time spent inside
                self.time_inside_sphere = (now_ns - self.point_in_sphere_start_time) / 1e9
                
                # Update max streak
                if self.time_inside_sphere > self.max_time_inside_sphere:
                    self.max_time_inside_sphere = self.time_inside_sphere
            else:
                # If we left the sphere, reset the streak
                self.point_in_sphere_start_time = None
                self.time_inside_sphere = 0.0

            # Throttle the compact log to 1 Hz
            t_now = time.monotonic()
            if t_now - self.last_compact_log_ts >= 1.0:
                self.last_compact_log_ts = t_now
                err_norm = math.sqrt(ex*ex + ey*ey + ez*ez)
                self.get_logger().info(
                    f"[GOAL {self.goal_counter}] "
                    f"Target=[{gx:.2f},{gy:.2f},{gz:.2f}] "
                    f"Err=[{ex:+.2f},{ey:+.2f},{ez:+.2f}] |‖e‖|={err_norm:.2f} "
                    f"In±0.4={in_sphere} streak={self.time_inside_sphere:.2f}s "
                    f"(max {self.max_time_inside_sphere:.2f}s)"
                )

    # ---------------- Action handlers ----------------
    def goal_callback(self, goal_request):
        '''
        Purpose:
        ---
        Callback executed when a new goal is received by the action server.
        It accepts all incoming goals.

        Input Arguments:
        ---
        `goal_request` : [ NavToWaypoint.Goal ]
            The goal request object sent by the client.

        Returns:
        ---
        `response` : [ rclpy.action.GoalResponse ]
            Returns `GoalResponse.ACCEPT` to accept the goal.

        Example call:
        ---
        (Called automatically by ROS 2 Action Server)
        '''
        self.get_logger().info('--- Received new goal request. Accepting. ---')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        '''
        Purpose:
        ---
        Callback executed when a goal cancellation is requested.
        It accepts the cancellation and sets `is_goal_active` to False.

        Input Arguments:
        ---
        `goal_handle` : [ rclpy.action.ActionServer.GoalHandle ]
            The handle to the goal that is being canceled.

        Returns:
        ---
        `response` : [ rclpy.action.CancelResponse ]
            Returns `CancelResponse.ACCEPT` to accept the cancellation.

        Example call:
        ---
        (Called automatically by ROS 2 Action Server)
        '''
        self.get_logger().info('Received cancel request. Accepting.')
        self.is_goal_active = False # Stop PID loop
        return CancelResponse.ACCEPT

    def execute_goal_callback(self, goal_handle):
        '''
        Purpose:
        ---
        The main execution function for an accepted goal. This function
        runs synchronously, blocking until the goal succeeds, is canceled,
        or aborts. It sets up PID targets, resets state, and monitors
        the drone's position for stabilization.

        Input Arguments:
        ---
        `goal_handle` : [ rclpy.action.ActionServer.GoalHandle ]
            The handle to the goal being executed.

        Returns:
        ---
        `result` : [ NavToWaypoint.Result ]
            The action result object, containing `time_taken`.

        Example call:
        ---
        (Called automatically by ROS 2 Action Server)
        '''
        
        # --- CAPTURE INITIAL YAW on first goal ---
        if not self.initial_yaw_set:
            self.yaw_target_degrees = self.yaw # self.yaw is continuously updated by odom_callback
            self.initial_yaw_set = True
            self.get_logger().warn(
                f"--- Captured Initial Yaw. Target Set to: {self.yaw_target_degrees:.2f} degrees ---"
            )
        
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

        # Z absolute -> relative for PID
        if self.base_height is None:
            self.get_logger().error("!!! Base height not calibrated yet! Cannot execute goal. Aborting. !!!")
            goal_handle.abort()
            return NavToWaypoint.Result()
        
        # Convert absolute Z goal to relative Z target for PID
        relative_goal_height = goal_z_abs - self.base_height
        target_z_relative = relative_goal_height 

        # Reset PID states, set new target
        self.get_logger().info("Resetting X, Y, Z PID states for new goal.")
        self.iterm_x, self.prev_error_x, self.filtered_derivative_x = 0.0, 0.0, 0.0
        self.iterm_y, self.prev_error_y, self.filtered_derivative_y = 0.0, 0.0, 0.0
        self.iterm_z, self.prev_error_z, self.filtered_derivative_z = 0.0, 0.0, 0.0
        # No yaw reset needed as there is no I or D term
        
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

        # 20 Hz blocking loop to monitor goal status
        loop_rate = self.create_rate(20)
        while rclpy.ok():
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.is_goal_active = False # Stop PID
                self.get_logger().info('Goal was canceled during execution.')
                return NavToWaypoint.Result()

            # Check for disarm
            if not self.armed:
                self.get_logger().warn("Drone disarmed during goal execution. Aborting.")
                goal_handle.abort()
                self.is_goal_active = False # Stop PID
                return NavToWaypoint.Result()

            # Publish feedback (PoseStamped)
            try:
                feedback_msg.current_pose.header = self.current_pose.header
                feedback_msg.current_pose.pose = self.current_pose.pose
                
                # --- HACK for Task 2A feedback (using time_inside_sphere) ---
                # The client code reads feedback.current_pose.header.stamp.sec
                # We populate that field with our max dwell time.
                if feedback_msg.current_pose.header.stamp.sec > 0:
                    # Overwrite 'sec' with our internal max dwell time
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

            # In-sphere check (using shared state updated in pid() loop)
            drone_in_sphere = (
                abs(err_x) < error_margin and
                abs(err_y) < error_margin and
                abs(err_z) < error_margin
            )

            # Success condition: 3 seconds of continuous stabilization
            if self.time_inside_sphere >= 3.0 and drone_in_sphere:
                self.get_logger().info(
                    f'>>> SENDING STATUS_SUCCEEDED for Goal #{self.goal_counter} '
                    f'(dwell={self.time_inside_sphere:.2f}s)'
                )
                stabilized = True
                break # Exit the while loop

            # Optional compact log from execute loop (throttled)
            self.get_logger().info(
                f"[GOAL {self.goal_counter}] "
                f"Target=[{goal_x:.2f},{goal_y:.2f},{goal_z_abs:.2f}] "
                f"Pos=[{current_x_abs:.2f},{current_y_abs:.2f},{current_z_abs:.2f}] "
                f"Err=[{err_x:+.2f},{err_y:+.2f},{err_z:+.2f}] |‖e‖|={err_norm:.2f} "
                f"In±0.4={drone_in_sphere} streak={self.time_inside_sphere:.2f}s "
                f"(max {self.max_time_inside_sphere:.2f}s)",
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

        # Deactivate PID, compute result
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
    Initializes the ROS 2 system, creates an instance of the WayPointServer,
    and spins a MultiThreadedExecutor to handle callbacks. Manages
    exceptions and ensures clean shutdown (disarming) on exit.

    Input Arguments:
    ---
    `args` : [ list, optional ]
        Command-line arguments passed to rclpy.init().

    Returns:
    ---
    None

    Example call:
    ---
    if __name__ == '__main__':
        main()
    '''
    rclpy.init(args=args)
    waypoint_server = None
    try:
        waypoint_server = WayPointServer()
        # Use MultiThreadedExecutor to allow callbacks to run in parallel
        executor = MultiThreadedExecutor()
        rclpy.spin(waypoint_server, executor=executor)
    except KeyboardInterrupt:
        pass # Handle Ctrl+C gracefully
    except Exception as e:
        if waypoint_server:
            waypoint_server.get_logger().error(f"Unhandled exception: {e}")
        else:
            print(f"Unhandled exception during node initialization: {e}")
    finally:
        # Ensure shutdown procedures are run
        if waypoint_server and rclpy.ok():
            waypoint_server.get_logger().info('Shutting down server node.')
            waypoint_server.disarm() # Disarm the drone on shutdown
            waypoint_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
