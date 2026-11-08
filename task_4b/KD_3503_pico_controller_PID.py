#!/usr/bin/env python3

'''
# Team ID:          3503
# Theme:            Swift Pico
# Author List:      Harsh Patnaik
# Filename:         KD_3503_pico_controller_PID.py
# Functions:        __init__, battery_callback, whycon_callback, start_arming, 
#                   complete_arming, disarm, pid_loop, write_forensic_log, main
# Global variables: None
'''

from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from geometry_msgs.msg import PoseArray
from crsf_msgs.msg import BatterySensor
import rclpy
from rclpy.node import Node
import time
import math
import os

class Swift_Pico_DampLimited_Steady(Node):
    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the PID controller node, sets up PID gains, state vectors, 
        ROS2 subscribers/publishers, and logging parameters.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        node = Swift_Pico_DampLimited_Steady()
        '''
        super().__init__('pico_controller')

        # --- SETPOINTS & LIMITS ---
        self.desired_state = [1.0, -4.5, 18.5] # target x, y, z coordinates
        self.latency_seconds = 0.100   
        self.tilt_limit = 100       # Max pitch/roll deviation from 1500
        self.thrust_overhead = 75      
        self.iterm_limit = 125
        self.deadband = 0.15           

        # --- DECOUPLED TUNING ---
        self.Kp_x, self.Kd_x, self.Ki_x = 12.5, 12.5, 0.2
        self.Kp_y, self.Kd_y, self.Ki_y = 32.5, 25.0, 0.2
        self.Kp_z, self.Kd_z, self.Ki_z = 15.0, 2.0, 0.0
        self.original_Kp_z = self.Kp_z  

        # --- STATE VECTORS ---
        self.est_pos = [0.0, 0.0, 0.0] # filtered position estimate
        self.filt_vel = [0.0, 0.0, 0.0] # filtered velocity estimate
        self.raw_whycon_pos = [0.0, 0.0, 0.0]
        self.alpha_pos = 0.8          # Alpha for position Low Pass Filter
        self.alpha_vel = 0.85         # Alpha for velocity Low Pass Filter          
        
        # --- INTERNAL STATE ---
        self.iterm_x, self.iterm_y, self.iterm_z = 0.0, 0.0, 0.0
        self.prev_error_x, self.prev_error_y, self.prev_error_z = 0.0, 0.0, 0.0
        self.filtered_derivative_x = 0.0
        self.filtered_derivative_y = 0.0
        self.filtered_derivative_z = 0.0
        self.last_whycon_time = time.time()
        self.last_loop_time = time.time()
        self.first_fix = False
        self.is_armed, self.is_landing = False, False
        self.stability_timer = 0.0
        self.outlier_count = 0
        self.loop_counter = 0
        self.whycon_msg_count = 0
        
        # --- SENSOR FREEZE DETECTION ---
        self.last_raw_pos = [0.0, 0.0, 0.0]
        self.freeze_count = 0
        
        # --- FORENSIC TRACKING VARIABLES ---
        self.prev_pos = [0.0, 0.0, 0.0]
        self.prev_vel = [0.0, 0.0, 0.0]
        self.accel = [0.0, 0.0, 0.0]
        self.accel_buffer = [[], [], []]
        self.vibration_rms = [0.0, 0.0, 0.0]
        self.raw_vel = [0.0, 0.0, 0.0]
        self.prev_d_val = [0.0, 0.0, 0.0]
        self.prev_pwm = [1500, 1500, 1500]
        self.exec_time = 0.0
        self.event_id = 0
        self.mission_phase = 0
        self.consec_sat = [0, 0, 0] 

        # --- POWER ---
        self.vbat = 4.35 # Battery voltage
        self.base_vbat, self.base_hover_value = 4.35, 1435
        self.hover_baseline = 1435
        
        # --- PWM RAMPING ---
        self.current_throttle_pwm = self.hover_baseline  
        self.last_ramp_time = time.time()
        self.ramp_interval = 0.1  
        self.ramp_step = 1  
        self.z_clamp_threshold = 4  
        self.z_floor_throttle = None
        self.prev_in_z_clamp_region = False
        self.z_clamp_applied = False  
        self.z_clamp_applied_at_z = None  
        self.z_clamp_hysteresis = 0.5  
        self.prev_err_z = 0.0  

        # --- LOGGING SETUP ---
        log_dir = os.path.expanduser('~/pico_ws2/src/swift_pico_hw/src')
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.log_path = os.path.join(log_dir, f"damp_limited_{int(time.time())}.csv")
        self.log_file = open(self.log_path, 'w')
        header = ("time,dt,freq,exec_time,vbat,hover_base,"
                  "raw_x,raw_y,raw_z,filt_x,filt_y,filt_z,pred_x,pred_y,pred_z,"
                  "raw_vx,raw_vy,raw_vz,filt_vx,filt_vy,filt_vz,"
                  "accel_x,accel_y,accel_z,vib_rms_x,vib_rms_y,vib_rms_z,"
                  "err_x,err_y,err_z,"
                  "target_vx,target_vy,drive_x,drive_y,damp_x,damp_y,"
                  "p_x,i_x,d_x,p_y,i_y,d_y,p_z,i_z,d_z,"
                  "Kp_z,Kd_z,Ki_z,"
                  "out_x,out_y,out_z,"
                  "pwm_pitch,pwm_roll,pwm_throt,"
                  "pwm_delta_pitch,pwm_delta_roll,pwm_delta_throt,"
                  "sat_pitch,sat_roll,sat_throt,consec_sat_pitch,consec_sat_roll,consec_sat_throt,"
                  "d_dot_x,d_dot_y,d_dot_z,"
                  "speed_2d,speed_mult,mission_phase,event_id,"
                  "outliers,freeze_count,is_armed\n")
        self.log_file.write(header)

        # --- ROS2 SETUP ---
        self.cmd = RCMessage()
        self.arm_service_client = self.create_client(CommandBool, '/drone/cmd/arming')
        self.command_pub = self.create_publisher(RCMessage, '/drone/rc_command', 10)
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(BatterySensor, '/drone/battery_info', self.battery_callback, 10)
        
        # Start the PID loop timer at 50Hz (0.02s)
        self.timer = self.create_timer(0.02, self.pid_loop)

    def battery_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the current battery voltage and recalculates the hover baseline 
        PWM to compensate for voltage drop.

        Input Arguments:
        ---
        `msg` :  [ BatterySensor ]
            The ROS2 message containing the current battery voltage (vbat).

        Returns:
        ---
        None

        Example call:
        ---
        Callback triggered by '/drone/battery_info' subscription.
        '''
        self.vbat = msg.vbat
        self.hover_baseline = int(self.base_hover_value + ((self.base_vbat - self.vbat) * 85))

    def whycon_callback(self, msg):
        '''
        Purpose:
        ---
        Processes incoming WhyCon position data, applies low-pass filtering, 
        calculates velocity, and detects sensor freezes/outliers.

        Input Arguments:
        ---
        `msg` :  [ PoseArray ]
            The ROS2 message containing detected WhyCon markers.

        Returns:
        ---
        None

        Example call:
        ---
        Callback triggered by '/whycon/poses' subscription.
        '''
        if not msg.poses: return
        self.whycon_msg_count += 1
        p = msg.poses[0].position
        now = time.time()
        dt_cam = now - self.last_whycon_time
        
        # Reinitialize if data was lost for more than 0.5 seconds
        if dt_cam > 0.5:
            self.get_logger().warn(f'WhyCon data gap detected ({dt_cam:.2f}s) - reinitializing')
            self.est_pos = [p.x, p.y, p.z]
            self.filt_vel = [0.0, 0.0, 0.0]
            self.last_raw_pos = [p.x, p.y, p.z]
            self.last_whycon_time = now
            self.iterm_x, self.iterm_y, self.iterm_z = 0.0, 0.0, 0.0
            return
        
        if not self.first_fix:
            self.est_pos = [p.x, p.y, p.z]
            self.last_raw_pos = [p.x, p.y, p.z]
            self.last_whycon_time = now
            self.first_fix = True
            self.start_arming()
            return

        # Position delta check for sensor freeze
        pos_delta = math.sqrt((p.x - self.last_raw_pos[0])**2 + 
                              (p.y - self.last_raw_pos[1])**2 + 
                              (p.z - self.last_raw_pos[2])**2)
        
        if pos_delta < 0.001:  # Position hasn't changed
            self.freeze_count += 1
            if self.freeze_count > 25: 
                self.get_logger().error('SENSOR FREEZE DETECTED - HOLDING HOVER')
                self.cmd.rc_throttle = self.hover_baseline
                self.cmd.rc_pitch = 1500
                self.cmd.rc_roll = 1500
                self.command_pub.publish(self.cmd)
                return
        else:
            self.freeze_count = 0
            self.last_raw_pos = [p.x, p.y, p.z]

        # Filter and update velocity and position
        if dt_cam > 0:
            self.raw_whycon_pos = [p.x, p.y, p.z]
            for i in range(3):
                p_val = self.raw_whycon_pos[i]
                prev_p = self.est_pos[i]
                self.est_pos[i] = (self.alpha_pos * p_val) + ((1 - self.alpha_pos) * self.est_pos[i])
                inst_vel = (self.est_pos[i] - prev_p) / dt_cam
                self.filt_vel[i] = (self.alpha_vel * inst_vel) + ((1 - self.alpha_vel) * self.filt_vel[i])
        
        self.last_whycon_time = now

    def start_arming(self):
        '''
        Purpose:
        ---
        Sends an asynchronous request to the arming service and sets a timer to 
        complete the process.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.start_arming()
        '''
        req = CommandBool.Request()
        req.value = True
        self.arm_service_client.call_async(req)
        self.arming_timer = self.create_timer(1.0, self.complete_arming)

    def complete_arming(self):
        '''
        Purpose:
        ---
        Finalizes the arming process, initializes the throttle, and cancels the 
        arming timer.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Triggered by arming_timer.
        '''
        if hasattr(self, 'arming_timer') and self.arming_timer is not None:
            self.arming_timer.cancel()
            self.arming_timer = None
        
        self.is_armed = True
        self.cmd.aux1 = 2000
        self.current_throttle_pwm = self.hover_baseline
        self.get_logger().info('Armed: Throttle set to hover baseline.')
    
    def disarm(self):
        '''
        Purpose:
        ---
        Sets all RC commands to safe values and marks the node as disarmed.

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
        self.is_armed = False
        self.cmd.rc_throttle = 1000
        self.cmd.rc_pitch = 1500
        self.cmd.rc_roll = 1500
        self.command_pub.publish(self.cmd)

    def pid_loop(self):
        '''
        Purpose:
        ---
        Main control loop that runs at 50Hz. Calculates PID outputs, handles 
        Z-axis clamping/ramping, and publishes RC messages to the drone.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Triggered by self.timer.
        '''
        start_time = time.time()
        if not self.is_armed or not self.first_fix: return
        now = time.time()
        dt = now - self.last_loop_time
        self.last_loop_time = now
        self.loop_counter += 1
        
        # Check for stale WhyCon data
        data_age = now - self.last_whycon_time
        if data_age > 0.5:
            self.cmd.rc_throttle = self.hover_baseline
            self.command_pub.publish(self.cmd)
            return

        # Calculate Position Error
        err_x = self.desired_state[0] - self.est_pos[0]
        err_y = self.desired_state[1] - self.est_pos[1]
        err_z = self.desired_state[2] - self.est_pos[2]

        # Implementation: Z-Clamp Logic to prevent overshooting during ascent
        self.in_z_clamp_region = abs(err_z) <= self.z_clamp_threshold
        approaching_from_below = (self.prev_err_z < -self.z_clamp_threshold and err_z >= -self.z_clamp_threshold)
        
        if self.in_z_clamp_region and approaching_from_below and not self.z_clamp_applied:
            self.z_floor_throttle = max(1000, self.current_throttle_pwm - 7)
            self.z_clamp_applied = True
            self.Kp_z = 2
            self.get_logger().info(f'Z CLAMP: Floor set at {self.z_floor_throttle}')

        # PID Terms Calculation
        self.iterm_x = max(-self.iterm_limit, min(self.iterm_limit, self.iterm_x + err_x * self.Ki_x * dt))
        self.iterm_y = max(-self.iterm_limit, min(self.iterm_limit, self.iterm_y + err_y * self.Ki_y * dt))
        
        # Derivative with filtering for X/Y
        raw_dx = (err_x - self.prev_error_x) / dt if dt > 0 else 0.0
        self.filtered_derivative_x = self.alpha_vel * self.filtered_derivative_x + (1 - self.alpha_vel) * raw_dx
        
        raw_dy = (err_y - self.prev_error_y) / dt if dt > 0 else 0.0
        self.filtered_derivative_y = self.alpha_vel * self.filtered_derivative_y + (1 - self.alpha_vel) * raw_dy

        out_x = (err_x * self.Kp_x) + self.iterm_x + (self.Kd_x * self.filtered_derivative_x)
        out_y = (err_y * self.Kp_y) + self.iterm_y + (self.Kd_y * self.filtered_derivative_y)
        out_z = (err_z * self.Kp_z) + self.iterm_z + (-self.Kd_z * self.filt_vel[2])

        # Final PWM Mapping
        self.cmd.rc_pitch = int(1500 + max(-self.tilt_limit, min(self.tilt_limit, out_x)))
        self.cmd.rc_roll = int(1500 + max(-self.tilt_limit, min(self.tilt_limit, out_y)))
        
        target_throttle = int(max(1100, min(self.hover_baseline + self.thrust_overhead, self.hover_baseline - out_z)))
        
        # PWM Ramping Logic
        if now - self.last_ramp_time >= self.ramp_interval:
            if self.current_throttle_pwm < target_throttle:
                self.current_throttle_pwm += self.ramp_step
            elif self.current_throttle_pwm > target_throttle:
                self.current_throttle_pwm -= self.ramp_step
            self.last_ramp_time = now
        
        # Enforce Z-floor safety
        if self.z_floor_throttle is not None and not self.is_landing:
            self.current_throttle_pwm = max(self.current_throttle_pwm, self.z_floor_throttle)
        
        self.cmd.rc_throttle = self.current_throttle_pwm
        self.command_pub.publish(self.cmd)
        
        # Update previous values for next iteration
        self.prev_error_x, self.prev_error_y, self.prev_err_z = err_x, err_y, err_z
        self.exec_time = time.time() - start_time
        
        # Forensic Logging
        self.write_forensic_log(now, dt, err_x, err_y, err_z, out_x, out_y, out_z)

    def write_forensic_log(self, now, dt, ex, ey, ez, ox, oy, oz):
        '''
        Purpose:
        ---
        Writes internal state data to a CSV file for flight analysis.

        Input Arguments:
        ---
        `now` :  [ float ]
            Current timestamp.
        `dt` :  [ float ]
            Time delta since last loop.
        `ex`, `ey`, `ez` : [ float ]
            Errors in x, y, z axes.
        `ox`, `oy`, `oz` : [ float ]
            PID outputs for pitch, roll, throttle.

        Returns:
        ---
        None

        Example call:
        ---
        self.write_forensic_log(now, dt, err_x, err_y, err_z, out_x, out_y, out_z)
        '''
        log_data = f"{now:.2f},{dt:.4f},{ex:.2f},{ey:.2f},{ez:.2f},{ox:.2f},{oy:.2f},{oz:.2f}\n"
        self.log_file.write(log_data)

def main():
    '''
    Purpose:
    ---
    Entry point for the ROS2 node. Initializes rclpy and spins the node.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    Called by the OS when the script is executed.
    '''
    rclpy.init()
    node = Swift_Pico_DampLimited_Steady()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: 
        node.disarm()
    finally:
        node.log_file.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
