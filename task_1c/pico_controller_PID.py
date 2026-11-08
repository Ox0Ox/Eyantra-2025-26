#!/usr/bin/env python3
'''
# Team ID:          eYRC#3503
# Theme:            Krishi Drone (KD)
# Author List:      <Harsh Patnaik
# Filename:         KD_3503_pico_controller.py
# Functions:        __init__, whycon_callback, arm, disarm, pid, main
# Global variables: None
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from swift_msgs.msg import SwiftMsgs
import time


class SwiftPicoPID(Node):
    '''
    Purpose:
    ---
    This class defines a 3D PID position controller node for the drone. 
    It subscribes to WhyCon pose estimates, computes control commands (Roll, Pitch, Throttle) 
    for all three axes (X, Y, Z), and publishes the corrected RC values for stable flight.
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the PID controller node, sets up publishers/subscribers,
        defines all PID constants for X, Y, and Z axes, initializes the 
        drone command message, and starts the control loop timer.
        '''
        super().__init__('pico_controller')

        # Publisher for sending control commands to the drone
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)

        # Subscriber for receiving positional feedback from WhyCon
        self.whycon_sub = self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 10)

        # Drone command message template
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000

        # --- Target Setpoint [x, y, z_relative] ---
        # Note: z_relative is based on your -14.0 logic
        self.target_pos = [-7.0, 0.0, -10.43] 

        # --- PID Tuning Parameters (Z-Axis / Throttle) --- (Your existing values)
        self.Kp_z = 30.0     # proportional gain
        self.Ki_z = 2.0     # integral gain
        self.Kd_z = 200.0     # derivative gain

        # --- PID Tuning Parameters (X-Axis / Roll) --- (Good starting values)
        self.Kp_x = 8.0
        self.Ki_x = 0.001
        self.Kd_x = 35.0

        # --- PID Tuning Parameters (Y-Axis / Pitch) --- (Good starting values)
        self.Kp_y = 8.0
        self.Ki_y = 0.001
        self.Kd_y = 35.0

        # --- PID State Variables (Z-Axis) --- (Your existing variables, renamed)
        self.iterm_z = 0.0
        self.iterm_limit_z = 100.0
        self.prev_error_z = 0.0
        self.current_pos_z = 0.0

        # --- PID State Variables (X-Axis) ---
        self.iterm_x = 0.0
        self.iterm_limit_x = 100.0
        self.prev_error_x = 0.0
        self.current_pos_x = 0.0
        
        # --- PID State Variables (Y-Axis) ---
        self.iterm_y = 0.0
        self.iterm_limit_y = 100.0
        self.prev_error_y = 0.0
        self.current_pos_y = 0.0

        # --- Other Control Variables ---
        self.sample_time = 0.06        # PID loop period in seconds
        self.alpha = 0.8               # smoothing factor for filtered feedback
        self.deadband = 0.0           # tolerance to ignore small oscillations
        self.base_height = None        # reference ground height
        self.armed = False             # flag to check drone state

        # Arm the drone and initialize PID loop
        self.arm()
        time.sleep(1)
        self.pid_timer = self.create_timer(self.sample_time, self.pid)
        self.get_logger().info("3D PID Controller Node Initialized ✅")

    def whycon_callback(self, msg):
        '''
        Purpose:
        ---
        Callback for WhyCon poses. Extracts X, Y, and Z positions, applies a 
        low-pass filter, and sets the ground reference height for Z-axis.
        '''
        if len(msg.poses) == 0:
            return

        pose = msg.poses[0]

        # --- Z-Axis Logic (Your existing logic) ---
        # Calibrate ground level only once
        if self.base_height is None:
            self.base_height = pose.position.z
            self.get_logger().info(f"Calibrated ground level at z = {self.base_height:.2f}")

        # Apply low-pass filtering for Z
        self.current_pos_z = self.alpha * self.current_pos_z + (1 - self.alpha) * pose.position.z
        
        # --- X-Axis Logic (New) ---
        self.current_pos_x = self.alpha * self.current_pos_x + (1 - self.alpha) * pose.position.x

        # --- Y-Axis Logic (New) ---
        self.current_pos_y = self.alpha * self.current_pos_y + (1 - self.alpha) * pose.position.y


    def arm(self):
        '''
        Purpose:
        ---
        Sends the necessary RC commands to arm the drone before flight control begins.
        '''
        self.get_logger().info("Arming Drone...")
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        time.sleep(0.2)

        # Neutralize all axes and arm
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.armed = True
        self.get_logger().info("Drone Armed ✅")

    def disarm(self):
        '''
        Purpose:
        ---
        Disarms the drone and sets throttle to minimum as a safety measure.
        '''
        self.get_logger().info("Disarming Drone...")
        self.cmd.rc_aux4 = 1000
        self.cmd.rc_throttle = 1000
        self.command_pub.publish(self.cmd)
        self.armed = False

    def pid(self):
        '''
        Purpose:
        ---
        Performs three independent PID control calculations (for X, Y, Z).
        Computes error for each axis, applies PID corrections, and publishes
        updated throttle, roll, and pitch commands.
        '''
        if not self.armed or self.base_height is None:
            self.get_logger().info("Waiting for first pose message...", throttle_duration_sec=1)
            return

        # --- Target Setpoints ---
        target_x = self.target_pos[0] 
        target_y = self.target_pos[1]
        target_z_relative = self.target_pos[2] # Your -14.0

        # -----------------------------------------------------------------
        # --- Z-AXIS (THROTTLE) CALCULATION (Your existing logic) ---
        # -----------------------------------------------------------------
        relative_position_z = self.current_pos_z - self.base_height
        if relative_position_z > 0:
            relative_position_z = 0.0

        error_z = target_z_relative - relative_position_z
        if abs(error_z) < self.deadband:
            error_z = 0.0

        derivative_z = (error_z - self.prev_error_z) / self.sample_time
        self.iterm_z += error_z * self.Ki_z
        self.iterm_z = max(-self.iterm_limit_z, min(self.iterm_limit_z, self.iterm_z))  # anti-windup
        self.iterm_z *= 0.98  # slow bleed-off

        p_term_z = self.Kp_z * error_z
        d_term_z = self.Kd_z * derivative_z
        pid_output_z = p_term_z + self.iterm_z + d_term_z

        throttle_value = int(1500 - pid_output_z) # Your mapping
        throttle_value = max(1000, min(2000, throttle_value))
        
        self.cmd.rc_throttle = throttle_value
        self.prev_error_z = error_z
        
        # -----------------------------------------------------------------
        # --- X-AXIS CALCULATION (Controls PITCH) ---
        # -----------------------------------------------------------------
        error_x = target_x - self.current_pos_x
        if abs(error_x) < self.deadband: # Apply deadband
            error_x = 0.0

        derivative_x = (error_x - self.prev_error_x) / self.sample_time
        self.iterm_x += error_x * self.Ki_x
        self.iterm_x = max(-self.iterm_limit_x, min(self.iterm_limit_x, self.iterm_x)) # Anti-windup

        p_term_x = self.Kp_x * error_x
        d_term_x = self.Kd_x * derivative_x
        pid_output_x = p_term_x + self.iterm_x + d_term_x

        # --- MAPPING TO PITCH ---
        # ** FIX: Swapping X-axis output to control PITCH **
        pitch_value = int(1500 + pid_output_x) 
        pitch_value = max(1000, min(2000, pitch_value)) # Clamp output

        self.cmd.rc_pitch = pitch_value # <-- Mapped to pitch
        self.prev_error_x = error_x

        # -----------------------------------------------------------------
        # --- Y-AXIS CALCULATION (Controls ROLL) ---
        # -----------------------------------------------------------------
        error_y = target_y - self.current_pos_y
        if abs(error_y) < self.deadband:
            error_y = 0.0

        derivative_y = (error_y - self.prev_error_y) / self.sample_time
        self.iterm_y += error_y * self.Ki_y
        self.iterm_y = max(-self.iterm_limit_y, min(self.iterm_limit_y, self.iterm_y))

        p_term_y = self.Kp_y * error_y
        d_term_y = self.Kd_y * derivative_y
        pid_output_y = p_term_y + self.iterm_y + d_term_y

        # --- MAPPING TO ROLL ---
        # ** FIX: Swapping Y-axis output to control ROLL **
        roll_value = int(1500 + pid_output_y)
        roll_value = max(1000, min(2000, roll_value))

        self.cmd.rc_roll = roll_value # <-- Mapped to roll
        self.prev_error_y = error_y

        # -----------------------------------------------------------------
        # --- Publish and Log ---
        # -----------------------------------------------------------------
        self.command_pub.publish(self.cmd) # Publish one message with all 3 commands

        self.get_logger().info(
            f"TARGET: [{target_x:.1f}, {target_y:.1f}, {target_z_relative:.1f}] | "
            f"POS: [X:{self.current_pos_x:.2f}, Y:{self.current_pos_y:.2f}, Z:{relative_position_z:.2f}] | "
            f"CMD: [R:{roll_value}, P:{pitch_value}, T:{throttle_value}]",
            throttle_duration_sec=0.2
        )


def main(args=None):
    '''
    Purpose:
    ---
    Initializes the ROS node, creates the PID controller object,
    and spins it until the program is terminated by the user.
    '''
    rclpy.init(args=args)
    controller = SwiftPicoPID()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.disarm()
    finally:
        controller.destroy_node()
        rclpy.shutdown()


# Function Name: main
#        Inputs: None
#       Outputs: None
#       Purpose: Launches the PID controller node and ensures safe shutdown.
if __name__ == '__main__':
    main()
