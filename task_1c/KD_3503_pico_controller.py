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
    This class defines a PID controller node for the drone. It subscribes to
    WhyCon pose estimates, computes control commands using PID principles,
    and publishes the corrected RC values for stable flight.
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the PID controller node, sets up the publishers and subscribers,
        defines all PID constants, initializes the drone command message, and
        starts the control loop timer.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        controller = SwiftPicoPID()
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

        # --- PID Tuning Parameters ---
        self.Kp = 28.0     # proportional gain
        self.Ki = 0.05     # integral gain
        self.Kd = 45.0     # derivative gain

        # --- PID State Variables ---
        self.iterm = 0.0               # integral term accumulator
        self.iterm_limit = 100.0       # clamp for integral windup
        self.prev_error = 0.0          # previous cycle error
        self.sample_time = 0.06        # PID loop period in seconds
        self.alpha = 0.8               # smoothing factor for filtered feedback

        # --- Setpoint and Tracking Variables ---
        self.target_altitude = -14.0   # desired position (in simulation coordinate)
        self.deadband = 0.25           # tolerance to ignore small oscillations
        self.base_height = None        # reference ground height
        self.current_position = 0.0    # filtered current position
        self.armed = False             # flag to check drone state

        # Arm the drone and initialize PID loop
        self.arm()
        time.sleep(1)
        self.pid_timer = self.create_timer(self.sample_time, self.pid)
        self.get_logger().info("PID Controller Node Initialized ✅")

    def whycon_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function for WhyCon pose messages. It extracts the current position
        of the drone, applies a low-pass filter for noise reduction, and sets the
        ground reference height on the first valid message.

        Input Arguments:
        ---
        `msg` :  [ geometry_msgs.msg.PoseArray ]
            Message containing the list of detected poses from the WhyCon system.

        Returns:
        ---
        None

        Example call:
        ---
        Automatically triggered when WhyCon publishes new pose data.
        '''
        if len(msg.poses) == 0:
            return

        pose = msg.poses[0]

        # Calibrate ground level only once
        if self.base_height is None:
            self.base_height = pose.position.z
            self.get_logger().info(f"Calibrated ground level at z = {self.base_height:.2f}")

        # Apply low-pass filtering for smoother readings
        self.current_position = self.alpha * self.current_position + (1 - self.alpha) * pose.position.z

    def arm(self):
        '''
        Purpose:
        ---
        Sends the necessary RC commands to arm the drone before flight control begins.

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
        self.command_pub.publish(self.cmd)
        self.armed = False

    def pid(self):
        '''
        Purpose:
        ---
        Performs the PID control logic for positional stabilization.
        Computes error between target and current position, applies PID
        corrections, and publishes updated throttle commands.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Called periodically by the internal ROS timer.
        '''
        if not self.armed or self.base_height is None:
            self.get_logger().info("Waiting for first pose message...", throttle_duration_sec=1)
            return

        # Compute relative position with respect to ground
        relative_position = self.current_position - self.base_height
        if relative_position > 0:
            relative_position = 0.0

        # Compute error and apply deadband
        error = self.target_altitude - relative_position
        if abs(error) < self.deadband:
            error = 0.0

        # PID calculations
        derivative = (error - self.prev_error) / self.sample_time
        self.iterm += error * self.Ki
        self.iterm = max(-self.iterm_limit, min(self.iterm_limit, self.iterm))  # anti-windup
        self.iterm *= 0.98  # slow bleed-off

        p_term = self.Kp * error
        d_term = self.Kd * derivative
        pid_output = p_term + self.iterm + d_term

        # Convert PID output to throttle command (1500 = hover)
        throttle_value = int(1500 - pid_output)
        throttle_value = max(1000, min(2000, throttle_value))

        # Update and publish drone command
        self.cmd.rc_throttle = throttle_value
        self.command_pub.publish(self.cmd)

        self.prev_error = error

        self.get_logger().info(
            f"MODE:STABILIZE | POS:[{relative_position:.2f}] | ERR:[{error:.2f}] | THR:{throttle_value}",
            throttle_duration_sec=0.2
        )


def main(args=None):
    '''
    Purpose:
    ---
    Initializes the ROS node, creates the PID controller object,
    and spins it until the program is terminated by the user.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    Automatically invoked when the script is executed.
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
