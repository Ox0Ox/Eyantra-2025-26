#!/usr/bin/env python3

'''
Team ID:          eYRC#3503
Control Strategy: MultiThreaded Two-Stage Hover-Window PID + Anti-Windup + Attitude Recorder
'''

import os
import shutil
import time
import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from geometry_msgs.msg import PoseArray
from crsf_msgs.msg import BatterySensor, Attitude


class Swift_Pico_DampLimited_Steady(Node):
    def __init__(self):
        super().__init__('pico_controller')

        self.node_start_time = time.time()
        self.arm_start_time = None

        # --- CALLBACK GROUPS & THREAD LOCK ---
        self.pid_callback_group = MutuallyExclusiveCallbackGroup()
        self.whycon_callback_group = ReentrantCallbackGroup()
        self.battery_callback_group = ReentrantCallbackGroup()
        self.arm_callback_group = MutuallyExclusiveCallbackGroup()
        self.state_lock = threading.Lock()

        # --- AXIS POLARITY ---
        self.invert_roll = False
        self.invert_pitch = False

        # --- SETPOINTS & LIMITS ---
        self.desired_state = [1.0, -4.5, 20.0]
        self.tilt_limit_pitch = 120    # Allows [1380, 1620] Pitch PWM
        self.tilt_limit_roll = 150     # Allows [1350, 1650] Roll PWM so Roll never clips at 1400
        self.thrust_overhead = 45      # Max climb overhead during initial takeoff stage
        self.hover_max_up = 28         # Increased from 22 -> 28 so Z centers on 20.0 instead of 20.69
        self.hover_max_down = 25       # Max PWM below hover_baseline once latched in hover zone
        self.iterm_limit = 25.0        # Anti-windup cap for X and Y
        self.iterm_limit_z = 22.0      # Anti-windup cap for Z
        self.deadband = 0.12

        # --- DECOUPLED TUNING ---
        self.Kp_x, self.Kd_x, self.Ki_x = 22.0, 26.0, 0.4
        self.Kp_y, self.Kd_y, self.Ki_y = 32.5, 26.0, 0.4
        self.Kp_z, self.Kd_z, self.Ki_z = 9.0, 22.0, 1.2
        self.original_Kp_z = self.Kp_z

        # --- ONBOARD ATTITUDE TELEMETRY (Passive Logging Only, rc_yaw = 1500) ---
        self.att_pitch_deg = 0.0
        self.att_roll_deg = 0.0
        self.att_yaw_deg = 0.0
        self.start_yaw_deg = None

        # --- STATE VECTORS & LOW-LAG Z FILTERS ---
        self.est_pos = [0.0, 0.0, 0.0]
        self.filt_vel = [0.0, 0.0, 0.0]
        self.raw_whycon_pos = [0.0, 0.0, 0.0]
        self.ground_z = None
        self.alpha_pos = 0.75          # XY position filter
        self.alpha_vel = 0.70          # XY velocity filter
        self.alpha_pos_z = 0.50        # Low-lag Z position filter (proven in Run 3617!)
        self.alpha_vel_z = 0.45        # Low-lag Z velocity filter (proven in Run 3617!)

        # --- INTERNAL STATE ---
        self.iterm_x, self.iterm_y, self.iterm_z = 0.0, 0.0, 0.0
        self.prev_error_x, self.prev_error_y, self.prev_error_z = 0.0, 0.0, 0.0
        self.last_whycon_time = time.time()
        self.last_dt_cam = 0.0
        self.new_cam_frame_flag = 0
        self.last_loop_time = time.time()
        self.first_fix = False
        self.is_armed, self.is_landing = False, False
        self.stability_timer = 0.0
        self.max_stability_timer = 0.0
        self.task4b_box_timer = 0.0
        self.max_task4b_box_timer = 0.0
        self.in_task4b_box_prev = False
        self.outlier_count = 0
        self.consec_outliers = 0
        self.loop_counter = 0
        self.whycon_msg_count = 0

        # --- SENSOR FREEZE DETECTION ---
        self.last_raw_pos = [0.0, 0.0, 0.0]
        self.freeze_count = 0
        self.max_freeze_count = 0
        self.stale_data_events = 0

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
        self.total_sat_ticks = [0, 0, 0]

        # --- SUMMARY STATISTICS TRACKERS ---
        self.vbat_start = None
        self.vbat_min = 99.0
        self.min_pos = [999.0, 999.0, 999.0]
        self.max_pos = [-999.0, -999.0, -999.0]
        self.max_abs_err = [0.0, 0.0, 0.0]
        self.sum_sq_err = [0.0, 0.0, 0.0]
        self.min_pwm = [2000, 2000, 2000]
        self.max_pwm = [1000, 1000, 1000]
        self.floor_enforcement_count = 0
        self.dt_min = 999.0
        self.dt_max = 0.0
        self.dt_cam_min = 999.0
        self.dt_cam_max = 0.0
        self.logs_closed = False

        # --- POWER ---
        self.vbat = 4.35
        self.base_vbat, self.base_hover_value = 4.35, 1500
        self.hover_baseline = 1500
        self.prev_hover_baseline = 1500

        # --- TWO-STAGE PWM RAMPING & HOVER WINDOW LATCH ---
        self.current_throttle_pwm = self.hover_baseline
        self.last_ramp_time = time.time()
        self.ramp_interval = 0.08          # Stage 1 (Takeoff): Gentle 12.5 PWM/s climb ramp
        self.hover_ramp_interval = 0.02    # Stage 2 (Hover Zone): 50Hz fast tracking inside safe [-25, +28] window
        self.ramp_step = 1
        self.hover_ramp_step = 3           # Up to 3 PWM per 0.02s tick (150 PWM/s) inside bounded hover window
        self.z_clamp_threshold = 2.5       # Latch hover window once within 2.5 units of target Z
        self.z_entry_throttle = None
        self.z_floor_throttle = None
        self.in_z_clamp_region = False
        self.prev_in_z_clamp_region = False
        self.z_clamp_applied = False
        self.prev_err_z = 0.0

        # --- DUAL LOGGING SETUP ---
        self.log_dir = os.path.expanduser('~/pico_ws2/src/swift_pico_hw/src')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        run_ts = int(time.time())
        self.csv_path = os.path.join(self.log_dir, f"damp_limited_{run_ts}.csv")
        self.latest_csv_path = os.path.join(self.log_dir, "latest_flight_telemetry.csv")
        self.txt_path = os.path.join(self.log_dir, f"flight_debug_{run_ts}.txt")
        self.latest_txt_path = os.path.join(self.log_dir, "latest_flight_debug.txt")

        self.log_file = open(self.csv_path, 'w')
        self.txt_file = open(self.txt_path, 'w')

        header = (
            "time,dt,freq,exec_time,vbat,hover_base,"
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
            "outliers,freeze_count,is_armed,"
            "t_rel,dt_cam,cam_age,new_cam,target_throt,z_floor,ff_throt,in_z_clamp,eff_Kp_z,stab_timer,is_landing,"
            "att_pitch_deg,att_roll_deg,att_yaw_deg\n"
        )
        self.log_file.write(header)

        self.txt_file.write("=================================================================\n")
        self.txt_file.write(f" SWIFT PICO HARDWARE FLIGHT DIAGNOSTIC LOG (Run ID: {run_ts})\n")
        self.txt_file.write("=================================================================\n")
        self.txt_file.write(f"Target Setpoint       : {self.desired_state}\n")
        self.txt_file.write(f"Axis Polarity         : invert_roll={self.invert_roll}, invert_pitch={self.invert_pitch}\n")
        self.txt_file.write(f"Base Hover PWM        : {self.base_hover_value} @ {self.base_vbat}V (Takeoff +{self.thrust_overhead} | Hover [-{self.hover_max_down}, +{self.hover_max_up}])\n")
        self.txt_file.write(f"PID X (Pitch)         : Kp={self.Kp_x}, Ki={self.Ki_x}, Kd={self.Kd_x} (limit={self.tilt_limit_pitch}, iterm_max={self.iterm_limit})\n")
        self.txt_file.write(f"PID Y (Roll)          : Kp={self.Kp_y}, Ki={self.Ki_y}, Kd={self.Kd_y} (limit={self.tilt_limit_roll}, iterm_max={self.iterm_limit})\n")
        self.txt_file.write(f"PID Z (Throttle)      : Kp={self.Kp_z}, Ki={self.Ki_z}, Kd={self.Kd_z}\n")
        self.txt_file.write(f"Z Filters             : alpha_pos_z={self.alpha_pos_z}, alpha_vel_z={self.alpha_vel_z}\n")
        self.txt_file.write(f"Two-Stage Ramping     : Takeoff=1/{self.ramp_interval}s | Hover={self.hover_ramp_step}/{self.hover_ramp_interval}s\n")
        self.txt_file.write("-----------------------------------------------------------------\n")
        self.txt_file.write(" CHRONOLOGICAL FLIGHT EVENTS\n")
        self.txt_file.write("-----------------------------------------------------------------\n")
        self.log_event("NODE_INIT", "Controller initialized. Waiting for first WhyCon frame and battery data.")

        # --- ROS2 SETUP ---
        self.cmd = RCMessage()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1000
        self.cmd.aux1 = 1000

        self.arm_service_client = self.create_client(
            CommandBool, '/drone/cmd/arming',
            callback_group=self.arm_callback_group
        )
        self.command_pub = self.create_publisher(RCMessage, '/drone/rc_command', 10)
        self.create_subscription(
            PoseArray, '/whycon/poses', self.whycon_callback, 1,
            callback_group=self.whycon_callback_group
        )
        self.create_subscription(
            BatterySensor, '/drone/battery_info', self.battery_callback, 10,
            callback_group=self.battery_callback_group
        )
        self.create_subscription(
            Attitude, '/drone/attitude', self.attitude_callback, 10,
            callback_group=self.battery_callback_group
        )

        self.timer = self.create_timer(
            0.02, self.pid_loop,
            callback_group=self.pid_callback_group
        )

    def log_event(self, tag, message):
        if self.logs_closed:
            return
        t_rel = time.time() - self.node_start_time
        line = f"[{t_rel:8.3f}s] [{tag:<18}] {message}\n"
        try:
            self.txt_file.write(line)
            self.txt_file.flush()
        except Exception:
            pass

    def attitude_callback(self, msg):
        with self.state_lock:
            self.att_pitch_deg = math.degrees(float(msg.pitch))
            self.att_roll_deg = math.degrees(float(msg.roll))
            self.att_yaw_deg = math.degrees(float(msg.yaw))
            if self.start_yaw_deg is None and self.first_fix:
                self.start_yaw_deg = self.att_yaw_deg
                self.log_event(
                    "ATTITUDE_INIT",
                    f"Initial FC Attitude: Pitch={self.att_pitch_deg:+.2f} deg, Roll={self.att_roll_deg:+.2f} deg, Yaw={self.att_yaw_deg:+.2f} deg"
                )

    def battery_callback(self, msg):
        with self.state_lock:
            self.vbat = msg.vbat
            new_hover = int(self.base_hover_value + ((self.base_vbat - self.vbat) * 85))
            if self.vbat_start is None and self.vbat > 1.0:
                self.vbat_start = self.vbat
                self.hover_baseline = new_hover
                self.prev_hover_baseline = new_hover
                self.log_event("BATTERY_INIT", f"Initial battery voltage: {self.vbat:.2f}V -> Hover Baseline: {self.hover_baseline} PWM")
            else:
                if self.is_armed and new_hover > self.prev_hover_baseline:
                    sag_comp = new_hover - self.prev_hover_baseline
                    self.current_throttle_pwm += sag_comp
                self.hover_baseline = new_hover
                self.prev_hover_baseline = new_hover

            if self.vbat > 1.0 and self.vbat < self.vbat_min:
                self.vbat_min = self.vbat

    def whycon_callback(self, msg):
        if not msg.poses:
            return

        p = msg.poses[0].position
        now = time.time()

        with self.state_lock:
            self.whycon_msg_count += 1
            dt_cam = now - self.last_whycon_time
            self.last_dt_cam = dt_cam
            self.new_cam_frame_flag = 1

            if not self.first_fix:
                self.raw_whycon_pos = [p.x, p.y, p.z]
                self.est_pos = [p.x, p.y, p.z]
                self.last_raw_pos = [p.x, p.y, p.z]
                self.ground_z = p.z
                self.prev_err_z = self.desired_state[2] - p.z
                self.last_whycon_time = now
                self.first_fix = True
                self.log_event(
                    "FIRST_WHYCON_FIX",
                    f"Initial pos: [{p.x:.2f}, {p.y:.2f}, {p.z:.2f}] | Ground Z={self.ground_z:.2f} | err_z={self.prev_err_z:.2f}. Arming..."
                )
                self.start_arming()
                return

            if 0.001 < dt_cam < 0.5:
                if dt_cam < self.dt_cam_min:
                    self.dt_cam_min = dt_cam
                if dt_cam > self.dt_cam_max:
                    self.dt_cam_max = dt_cam

            if dt_cam > 0.5:
                self.get_logger().warn(f'WhyCon data gap detected ({dt_cam:.2f}s) - reinitializing velocity')
                self.log_event("WHYCON_DATA_GAP", f"Camera gap of {dt_cam:.3f}s detected! New pos: [{p.x:.2f}, {p.y:.2f}, {p.z:.2f}]")
                self.raw_whycon_pos = [p.x, p.y, p.z]
                self.est_pos = [p.x, p.y, p.z]
                self.filt_vel = [0.0, 0.0, 0.0]
                self.raw_vel = [0.0, 0.0, 0.0]
                self.last_raw_pos = [p.x, p.y, p.z]
                self.consec_outliers = 0
                self.last_whycon_time = now
                return

            pos_delta = math.sqrt(
                (p.x - self.last_raw_pos[0])**2 +
                (p.y - self.last_raw_pos[1])**2 +
                (p.z - self.last_raw_pos[2])**2
            )

            if pos_delta == 0.0:
                self.freeze_count += 1
                if self.freeze_count > self.max_freeze_count:
                    self.max_freeze_count = self.freeze_count
                if self.freeze_count == 26:
                    self.log_event("SENSOR_FREEZE", f"Position identical for >25 frames at {self.last_raw_pos}. Holding throttle={self.current_throttle_pwm}")
                if self.freeze_count > 25:
                    self.filt_vel = [0.0, 0.0, 0.0]
                    self.cmd.rc_throttle = self.current_throttle_pwm
                    self.cmd.rc_pitch = 1500
                    self.cmd.rc_roll = 1500
                    self.cmd.rc_yaw = 1500
                    self.command_pub.publish(self.cmd)
                    return
            else:
                self.freeze_count = 0

            dist_jump = math.sqrt((p.x - self.est_pos[0])**2 + (p.y - self.est_pos[1])**2)
            if dist_jump > 1.5 and self.consec_outliers < 3:
                self.outlier_count += 1
                self.consec_outliers += 1
                self.log_event(
                    "OUTLIER_REJECTED",
                    f"Rejected XY jump of {dist_jump:.2f} (#{self.consec_outliers}) (raw=[{p.x:.2f},{p.y:.2f},{p.z:.2f}] vs est=[{self.est_pos[0]:.2f},{self.est_pos[1]:.2f},{self.est_pos[2]:.2f}])"
                )
                return
            elif dist_jump > 1.5 and self.consec_outliers >= 3:
                self.log_event("OUTLIER_RESET", f"Accepted new XY position [{p.x:.2f}, {p.y:.2f}] after 3 consecutive jumps.")
                self.est_pos = [p.x, p.y, p.z]
                self.filt_vel = [0.0, 0.0, 0.0]
                self.consec_outliers = 0
            else:
                self.consec_outliers = 0

            if dt_cam > 1e-4:
                new_raw = [p.x, p.y, p.z]
                for i in range(2):
                    self.raw_vel[i] = (new_raw[i] - self.last_raw_pos[i]) / dt_cam
                    prev_p = self.est_pos[i]
                    self.est_pos[i] = (self.alpha_pos * new_raw[i]) + ((1.0 - self.alpha_pos) * prev_p)
                    inst_vel = (self.est_pos[i] - prev_p) / dt_cam
                    self.filt_vel[i] = (self.alpha_vel * inst_vel) + ((1.0 - self.alpha_vel) * self.filt_vel[i])

                self.raw_vel[2] = (new_raw[2] - self.last_raw_pos[2]) / dt_cam
                prev_pz = self.est_pos[2]
                self.est_pos[2] = (self.alpha_pos_z * new_raw[2]) + ((1.0 - self.alpha_pos_z) * prev_pz)
                inst_vz = (self.est_pos[2] - prev_pz) / dt_cam
                self.filt_vel[2] = (self.alpha_vel_z * inst_vz) + ((1.0 - self.alpha_vel_z) * self.filt_vel[2])

                self.raw_whycon_pos = new_raw
                self.last_raw_pos = list(new_raw)

            self.last_whycon_time = now

    def start_arming(self):
        req = CommandBool.Request()
        req.value = True
        self.arm_service_client.call_async(req)
        self.arming_timer = self.create_timer(
            1.0, self.complete_arming,
            callback_group=self.arm_callback_group
        )

    def complete_arming(self):
        if hasattr(self, 'arming_timer') and self.arming_timer is not None:
            self.arming_timer.cancel()
            self.arming_timer = None

        with self.state_lock:
            self.is_armed = True
            self.arm_start_time = time.time()
            self.last_loop_time = time.time()
            self.last_ramp_time = time.time()
            self.cmd.aux1 = 2000
            self.cmd.rc_roll = 1500
            self.cmd.rc_pitch = 1500
            self.cmd.rc_yaw = 1500
            self.current_throttle_pwm = self.hover_baseline
            self.prev_hover_baseline = self.hover_baseline
            self.log_event(
                "DRONE_ARMED",
                f"Armed! Starting throttle at hover_baseline={self.hover_baseline} PWM (vbat={self.vbat:.2f}V)"
            )
        self.get_logger().info(
            f'Armed: Starting throttle at battery-compensated hover={self.hover_baseline} (vbat={self.vbat:.2f}V)'
        )

    def disarm(self):
        with self.state_lock:
            self.is_armed = False
            self.cmd.rc_throttle = 1000
            self.cmd.rc_pitch = 1500
            self.cmd.rc_roll = 1500
            self.cmd.rc_yaw = 1500
            self.cmd.aux1 = 1000
            self.command_pub.publish(self.cmd)
            self.log_event("DRONE_DISARMED", "Disarm command sent (throttle=1000, aux1=1000).")

    def pid_loop(self):
        start_time = time.time()

        with self.state_lock:
            if not self.is_armed or not self.first_fix:
                return

            now = time.time()
            t_rel = now - self.node_start_time
            dt = now - self.last_loop_time
            self.last_loop_time = now
            if dt <= 0.0 or dt > 0.2:
                dt = 0.02
            else:
                if dt < self.dt_min:
                    self.dt_min = dt
                if dt > self.dt_max:
                    self.dt_max = dt

            self.loop_counter += 1
            self.event_id = 0

            data_age = now - self.last_whycon_time
            if data_age > 0.5:
                self.stale_data_events += 1
                safe_hold_throt = min(self.current_throttle_pwm, self.hover_baseline)
                if self.stale_data_events == 1 or self.stale_data_events % 25 == 0:
                    self.log_event("STALE_WHYCON", f"No WhyCon frame for {data_age:.2f}s! Safe holding throttle={safe_hold_throt}")
                self.get_logger().error(
                    f'STALE WHYCON DATA ({data_age:.2f}s old) - HOLDING HOVER BASELINE ({safe_hold_throt})',
                    throttle_duration_sec=1.0
                )
                self.cmd.rc_throttle = safe_hold_throt
                self.cmd.rc_pitch = 1500
                self.cmd.rc_roll = 1500
                self.cmd.rc_yaw = 1500
                self.command_pub.publish(self.cmd)
                return

            est_pos = list(self.est_pos)
            filt_vel = list(self.filt_vel)
            raw_whycon_pos = list(self.raw_whycon_pos)
            raw_vel = list(self.raw_vel)
            hover_baseline = self.hover_baseline
            vbat = self.vbat
            outlier_count = self.outlier_count
            freeze_count = self.freeze_count
            is_armed = self.is_armed
            dt_cam = self.last_dt_cam
            new_cam = self.new_cam_frame_flag
            self.new_cam_frame_flag = 0
            att_p = self.att_pitch_deg
            att_r = self.att_roll_deg
            att_y = self.att_yaw_deg

        for i in range(3):
            self.accel[i] = (filt_vel[i] - self.prev_vel[i]) / dt if dt > 0 else 0.0
            self.accel_buffer[i].append(self.accel[i])
            if len(self.accel_buffer[i]) > 20:
                self.accel_buffer[i].pop(0)
            if len(self.accel_buffer[i]) > 0:
                self.vibration_rms[i] = math.sqrt(
                    sum(a**2 for a in self.accel_buffer[i]) / len(self.accel_buffer[i])
                )

        raw_err_dist = math.sqrt(sum((self.desired_state[i] - raw_whycon_pos[i])**2 for i in range(3)))
        if raw_err_dist > 5.0:
            self.mission_phase = 1
        elif raw_err_dist > 0.8:
            self.mission_phase = 2
        else:
            self.mission_phase = 3

        # 1. Position Error
        raw_err_x = self.desired_state[0] - est_pos[0]
        raw_err_y = self.desired_state[1] - est_pos[1]
        err_z = self.desired_state[2] - est_pos[2]

        raw_errs = [raw_err_x, raw_err_y, err_z]
        for i in range(3):
            if est_pos[i] < self.min_pos[i]:
                self.min_pos[i] = est_pos[i]
            if est_pos[i] > self.max_pos[i]:
                self.max_pos[i] = est_pos[i]
            if abs(raw_errs[i]) > self.max_abs_err[i]:
                self.max_abs_err[i] = abs(raw_errs[i])
            self.sum_sq_err[i] += raw_errs[i] ** 2

        err_x = 0.0 if abs(raw_err_x) < self.deadband else raw_err_x
        err_y = 0.0 if abs(raw_err_y) < self.deadband else raw_err_y
        err_z_db = 0.0 if abs(err_z) < self.deadband else err_z

        # --- TWO-STAGE HOVER WINDOW LATCH ---
        self.in_z_clamp_region = abs(err_z) <= self.z_clamp_threshold
        if self.in_z_clamp_region and not self.z_clamp_applied:
            self.z_entry_throttle = self.current_throttle_pwm
            clamp_target = hover_baseline + 5
            self.current_throttle_pwm = clamp_target
            self.z_clamp_applied = True
            self.event_id = 1
            msg_str = (
                f'First entry into hover band at err_z={err_z:.2f}! '
                f'Entry throt={self.z_entry_throttle} -> Reset to equilibrium {clamp_target}. '
                f'Latched hover window [{hover_baseline - self.hover_max_down}, {hover_baseline + self.hover_max_up}]'
            )
            self.log_event("HOVER_WINDOW_LATCH", msg_str)
            self.get_logger().info(f'HOVER WINDOW LATCHED: {msg_str}')

        self.prev_in_z_clamp_region = self.in_z_clamp_region
        self.prev_err_z = err_z

        # 2. Core PID Logic with Conditional Anti-Windup
        p_x = err_x * self.Kp_x
        d_x = -self.Kd_x * filt_vel[0]
        p_y = err_y * self.Kp_y
        d_y = -self.Kd_y * filt_vel[1]

        is_airborne = (self.ground_z is not None) and ((self.ground_z - est_pos[2]) > 0.8)
        if is_airborne or self.z_clamp_applied:
            tentative_out_x = p_x + self.iterm_x + d_x
            if abs(tentative_out_x) < self.tilt_limit_pitch or (tentative_out_x * err_x < 0):
                self.iterm_x = max(-self.iterm_limit, min(self.iterm_limit, self.iterm_x + err_x * self.Ki_x * dt))

            tentative_out_y = p_y + self.iterm_y + d_y
            if abs(tentative_out_y) < self.tilt_limit_roll or (tentative_out_y * err_y < 0):
                self.iterm_y = max(-self.iterm_limit, min(self.iterm_limit, self.iterm_y + err_y * self.Ki_y * dt))

        if self.z_clamp_applied:
            self.iterm_z = max(-self.iterm_limit_z, min(self.iterm_limit_z, self.iterm_z + err_z_db * self.Ki_z * dt))

        i_x = self.iterm_x
        i_y = self.iterm_y
        self.prev_error_x = err_x
        self.prev_error_y = err_y

        effective_Kp_z = self.Kp_z
        p_z = err_z_db * effective_Kp_z
        i_z = self.iterm_z
        raw_dz = -self.Kd_z * filt_vel[2]
        d_z = max(-100.0, min(100.0, raw_dz))
        self.prev_error_z = err_z

        out_x = p_x + i_x + d_x
        out_y = p_y + i_y + d_y
        out_z = p_z + i_z + d_z

        target_vx = p_x
        target_vy = p_y
        drive_x = p_x + i_x
        drive_y = p_y + i_y
        damp_x = d_x
        damp_y = d_y

        d_dot = [
            (d_x - self.prev_d_val[0]) / dt if dt > 0 else 0.0,
            (d_y - self.prev_d_val[1]) / dt if dt > 0 else 0.0,
            (d_z - self.prev_d_val[2]) / dt if dt > 0 else 0.0
        ]
        self.prev_d_val = [d_x, d_y, d_z]

        # 4. Final Saturations (Pitch, Roll, and Neutral Yaw=1500)
        pitch_step = max(-self.tilt_limit_pitch, min(self.tilt_limit_pitch, out_x))
        roll_step = max(-self.tilt_limit_roll, min(self.tilt_limit_roll, out_y))

        self.cmd.rc_pitch = int(1500 - pitch_step) if self.invert_pitch else int(1500 + pitch_step)
        self.cmd.rc_roll = int(1500 - roll_step) if self.invert_roll else int(1500 + roll_step)
        self.cmd.rc_yaw = 1500

        # Stage-dependent Throttle Bounds & Ramp Speed:
        if not self.z_clamp_applied:
            max_th = hover_baseline + self.thrust_overhead
            min_th = hover_baseline - 15
            self.z_floor_throttle = None
            target_throttle = int(max(min_th, min(max_th, hover_baseline - out_z)))
            current_ramp_interval = self.ramp_interval
            step_size = self.ramp_step
        else:
            max_th = hover_baseline + self.hover_max_up
            min_th = hover_baseline - self.hover_max_down
            self.z_floor_throttle = min_th
            target_throttle = int(max(min_th, min(max_th, hover_baseline - out_z)))
            current_ramp_interval = self.hover_ramp_interval
            step_size = self.hover_ramp_step

        time_since_last_ramp = now - self.last_ramp_time
        if time_since_last_ramp >= current_ramp_interval:
            steps = max(1, int(time_since_last_ramp / current_ramp_interval))
            max_delta = step_size * steps
            if self.current_throttle_pwm < target_throttle:
                self.current_throttle_pwm = min(target_throttle, self.current_throttle_pwm + max_delta)
            elif self.current_throttle_pwm > target_throttle:
                self.current_throttle_pwm = max(target_throttle, self.current_throttle_pwm - max_delta)
            self.last_ramp_time = now

        self.current_throttle_pwm = max(min_th, min(max_th, self.current_throttle_pwm))
        self.cmd.rc_throttle = self.current_throttle_pwm

        # Feedforward: Add 2-6 PWM when tilted, allowed up to max_th + 6 so tilt never causes altitude sag
        roll_mag = abs(self.cmd.rc_roll - 1500)
        pitch_mag = abs(self.cmd.rc_pitch - 1500)
        tilt_magnitude = roll_mag + pitch_mag
        ff_throttle = 0
        if tilt_magnitude > 30 and err_z <= 1.0:
            ff_throttle = int(2 + max(0.0, min(4.0, (tilt_magnitude - 30) * 4.0 / 150.0)))
            self.cmd.rc_throttle = min(max_th + 6, self.cmd.rc_throttle + ff_throttle)

        sat_pitch = 1 if abs(out_x) >= self.tilt_limit_pitch else 0
        sat_roll = 1 if abs(out_y) >= self.tilt_limit_roll else 0
        sat_throt = 1 if (hover_baseline - out_z) >= max_th else 0

        self.total_sat_ticks[0] += sat_pitch
        self.total_sat_ticks[1] += sat_roll
        self.total_sat_ticks[2] += sat_throt

        self.consec_sat[0] = (self.consec_sat[0] + 1) if sat_pitch else 0
        self.consec_sat[1] = (self.consec_sat[1] + 1) if sat_roll else 0
        self.consec_sat[2] = (self.consec_sat[2] + 1) if sat_throt else 0

        if self.consec_sat[0] == 25:
            self.log_event("PITCH_SATURATED", f"Pitch saturated at {self.cmd.rc_pitch} for 0.5s (err_x={raw_err_x:.2f}, vel_x={filt_vel[0]:.2f})")
        if self.consec_sat[1] == 25:
            self.log_event("ROLL_SATURATED", f"Roll saturated at {self.cmd.rc_roll} for 0.5s (err_y={raw_err_y:.2f}, vel_y={filt_vel[1]:.2f})")

        pwm_delta = [
            self.cmd.rc_pitch - self.prev_pwm[0],
            self.cmd.rc_roll - self.prev_pwm[1],
            self.cmd.rc_throttle - self.prev_pwm[2]
        ]
        self.prev_pwm = [self.cmd.rc_pitch, self.cmd.rc_roll, self.cmd.rc_throttle]

        # Track Task 4B Official Box (±0.8 in X, Y, Z)
        in_task4b_box = (abs(raw_err_x) <= 0.8 and abs(raw_err_y) <= 0.8 and abs(err_z) <= 0.8)
        if in_task4b_box:
            self.task4b_box_timer += dt
            if self.task4b_box_timer > self.max_task4b_box_timer:
                self.max_task4b_box_timer = self.task4b_box_timer
            if not self.in_task4b_box_prev:
                self.log_event("ENTER_0.8_BOX", f"Entered ±0.8 box! Pos=[{est_pos[0]:.2f},{est_pos[1]:.2f},{est_pos[2]:.2f}]")
        else:
            if self.in_task4b_box_prev:
                self.log_event("EXIT_0.8_BOX", f"Exited ±0.8 box after {self.task4b_box_timer:.2f}s. Err=[{raw_err_x:+.2f},{raw_err_y:+.2f},{err_z:+.2f}]")
            self.task4b_box_timer = 0.0
        self.in_task4b_box_prev = in_task4b_box

        # 5. Stability & Landing logic (Trigger landing after 10.5s continuous dwell in ±0.8 box)
        if self.task4b_box_timer > 10.5 and not self.is_landing:
            self.is_landing = True
            self.event_id = 2
            self.log_event("AUTO_LAND_START", "10.5s continuous dwell achieved in ±0.8 box! Initiating smooth landing.")

        dist = math.sqrt(raw_err_x**2 + raw_err_y**2 + err_z**2)
        if dist < 0.5:
            self.stability_timer += dt
            if self.stability_timer > self.max_stability_timer:
                self.max_stability_timer = self.stability_timer
        else:
            self.stability_timer = 0.0

        if self.is_landing:
            target_landing_pwm = hover_baseline - 35
            if self.current_throttle_pwm > target_landing_pwm:
                self.current_throttle_pwm = max(target_landing_pwm, self.current_throttle_pwm - 1)
            self.cmd.rc_throttle = self.current_throttle_pwm
            self.cmd.rc_pitch, self.cmd.rc_roll = 1500, 1500

        pwms = [self.cmd.rc_pitch, self.cmd.rc_roll, self.cmd.rc_throttle]
        for i in range(3):
            if pwms[i] < self.min_pwm[i]:
                self.min_pwm[i] = pwms[i]
            if pwms[i] > self.max_pwm[i]:
                self.max_pwm[i] = pwms[i]

        self.command_pub.publish(self.cmd)

        self.get_logger().info(
            f"[{t_rel:5.1f}s] Pos:[{est_pos[0]:+.2f},{est_pos[1]:+.2f},{est_pos[2]:.2f}] "
            f"Err:[{raw_err_x:+.2f},{raw_err_y:+.2f},{err_z:+.2f}] "
            f"PWM:[P:{self.cmd.rc_pitch},R:{self.cmd.rc_roll},T:{self.cmd.rc_throttle}(tgt:{target_throttle})] "
            f"Box0.8:{self.task4b_box_timer:.1f}s Vbat:{vbat:.2f}V",
            throttle_duration_sec=1.0
        )

        self.prev_pos = list(est_pos)
        self.prev_vel = list(filt_vel)
        self.exec_time = time.time() - start_time

        self.write_forensic_log(
            now, t_rel, dt, dt_cam, data_age, new_cam,
            vbat, hover_baseline, raw_whycon_pos, est_pos, raw_vel, filt_vel,
            [raw_err_x, raw_err_y, err_z],
            [target_vx, target_vy], [drive_x, drive_y], [damp_x, damp_y],
            [p_x, i_x, d_x], [p_y, i_y, d_y], [p_z, i_z, d_z],
            [out_x, out_y, out_z], d_dot, pwm_delta,
            [sat_pitch, sat_roll, sat_throt],
            target_throttle, ff_throttle, effective_Kp_z,
            outlier_count, freeze_count, is_armed,
            att_p, att_r, att_y
        )

    def write_forensic_log(
        self, now, t_rel, dt, dt_cam, data_age, new_cam,
        vbat, hover_baseline, raw_whycon_pos, est_pos, raw_vel, filt_vel,
        err, target_v, drive, damp, pid_x, pid_y, pid_z, out, d_dot, pwm_delta, sat,
        target_throt, ff_throt, eff_Kp_z,
        outlier_count, freeze_count, is_armed,
        att_p, att_r, att_y
    ):
        if self.logs_closed:
            return
        freq = 1.0 / dt if dt > 0 else 0.0
        speed_2d = math.sqrt(filt_vel[0]**2 + filt_vel[1]**2)
        z_flr = self.z_floor_throttle if self.z_floor_throttle is not None else 0
        log_data = (
            f"{now:.3f},{dt:.4f},{freq:.1f},{self.exec_time:.6f},"
            f"{vbat:.2f},{hover_baseline},"
            f"{raw_whycon_pos[0]:.3f},{raw_whycon_pos[1]:.3f},{raw_whycon_pos[2]:.3f},"
            f"{est_pos[0]:.3f},{est_pos[1]:.3f},{est_pos[2]:.3f},"
            f"{est_pos[0]:.3f},{est_pos[1]:.3f},{est_pos[2]:.3f},"
            f"{raw_vel[0]:.3f},{raw_vel[1]:.3f},{raw_vel[2]:.3f},"
            f"{filt_vel[0]:.3f},{filt_vel[1]:.3f},{filt_vel[2]:.3f},"
            f"{self.accel[0]:.2f},{self.accel[1]:.2f},{self.accel[2]:.2f},"
            f"{self.vibration_rms[0]:.4f},{self.vibration_rms[1]:.4f},{self.vibration_rms[2]:.4f},"
            f"{err[0]:.3f},{err[1]:.3f},{err[2]:.3f},"
            f"{target_v[0]:.2f},{target_v[1]:.2f},{drive[0]:.2f},{drive[1]:.2f},{damp[0]:.2f},{damp[1]:.2f},"
            f"{pid_x[0]:.2f},{pid_x[1]:.2f},{pid_x[2]:.2f},"
            f"{pid_y[0]:.2f},{pid_y[1]:.2f},{pid_y[2]:.2f},"
            f"{pid_z[0]:.2f},{pid_z[1]:.2f},{pid_z[2]:.2f},"
            f"{self.Kp_z:.2f},{self.Kd_z:.2f},{self.Ki_z:.2f},"
            f"{out[0]:.2f},{out[1]:.2f},{out_z_val(out[2]):.2f},"
            f"{self.cmd.rc_pitch},{self.cmd.rc_roll},{self.cmd.rc_throttle},"
            f"{pwm_delta[0]},{pwm_delta[1]},{pwm_delta[2]},"
            f"{sat[0]},{sat[1]},{sat[2]},{self.consec_sat[0]},{self.consec_sat[1]},{self.consec_sat[2]},"
            f"{d_dot[0]:.4f},{d_dot[1]:.4f},{d_dot[2]:.4f},"
            f"{speed_2d:.3f},1.0,{self.mission_phase},{self.event_id},"
            f"{outlier_count},{freeze_count},{1 if is_armed else 0},"
            f"{t_rel:.4f},{dt_cam:.4f},{data_age:.4f},{new_cam},"
            f"{target_throt},{z_flr},{ff_throt},{1 if self.in_z_clamp_region else 0},"
            f"{eff_Kp_z:.2f},{self.stability_timer:.2f},{1 if self.is_landing else 0},"
            f"{att_p:.2f},{att_r:.2f},{att_y:.2f}\n"
        )
        self.log_file.write(log_data)
        if self.loop_counter % 50 == 0:
            self.log_file.flush()

    def close_logs(self):
        if self.logs_closed:
            return
        self.logs_closed = True

        total_flight_sec = (time.time() - self.arm_start_time) if self.arm_start_time else 0.0
        n_ticks = max(1, self.loop_counter)
        rms_err = [math.sqrt(self.sum_sq_err[i] / n_ticks) for i in range(3)]
        avg_pid_hz = self.loop_counter / total_flight_sec if total_flight_sec > 0 else 0.0
        avg_cam_hz = self.whycon_msg_count / (time.time() - self.node_start_time)

        try:
            self.txt_file.write("\n=================================================================\n")
            self.txt_file.write(" POST-FLIGHT AUTOMATED DIAGNOSTIC SUMMARY\n")
            self.txt_file.write("=================================================================\n")
            self.txt_file.write(f"Total Armed Flight Time  : {total_flight_sec:.2f} s ({self.loop_counter} PID ticks)\n")
            self.txt_file.write(f"Average PID Loop Rate    : {avg_pid_hz:.1f} Hz (dt min={self.dt_min:.4f}s, max={self.dt_max:.4f}s)\n")
            self.txt_file.write(f"WhyCon Camera Frames     : {self.whycon_msg_count} frames ({avg_cam_hz:.1f} FPS | dt_cam min={self.dt_cam_min:.3f}s, max={self.dt_cam_max:.3f}s)\n")
            self.txt_file.write(f"Camera Anomalies         : Outliers={self.outlier_count}, MaxFreeze={self.max_freeze_count} frames, StaleTicks={self.stale_data_events}\n")
            self.txt_file.write(f"Battery & Hover Baseline : Start={self.vbat_start if self.vbat_start else self.vbat:.2f}V, Min={self.vbat_min:.2f}V, End={self.vbat:.2f}V (Final Hover Base={self.hover_baseline})\n")
            self.txt_file.write(f"FC Attitude Start -> End : Start Yaw={self.start_yaw_deg if self.start_yaw_deg is not None else 0.0:.2f} deg -> Final [P:{self.att_pitch_deg:+.2f}, R:{self.att_roll_deg:+.2f}, Y:{self.att_yaw_deg:+.2f}] deg\n")
            self.txt_file.write("-----------------------------------------------------------------\n")
            self.txt_file.write(f"Position Range X         : [{self.min_pos[0]:.2f}, {self.max_pos[0]:.2f}] (Target: {self.desired_state[0]}) | RMS Err: {rms_err[0]:.2f} | Max Abs Err: {self.max_abs_err[0]:.2f}\n")
            self.txt_file.write(f"Position Range Y         : [{self.min_pos[1]:.2f}, {self.max_pos[1]:.2f}] (Target: {self.desired_state[1]}) | RMS Err: {rms_err[1]:.2f} | Max Abs Err: {self.max_abs_err[1]:.2f}\n")
            self.txt_file.write(f"Position Range Z         : [{self.min_pos[2]:.2f}, {self.max_pos[2]:.2f}] (Target: {self.desired_state[2]}) | RMS Err: {rms_err[2]:.2f} | Max Abs Err: {self.max_abs_err[2]:.2f}\n")
            self.txt_file.write(f"Final I-Terms            : iterm_x={self.iterm_x:.2f}, iterm_y={self.iterm_y:.2f}, iterm_z={self.iterm_z:.2f}\n")
            self.txt_file.write("-----------------------------------------------------------------\n")
            self.txt_file.write(f"PWM Pitch Range          : [{self.min_pwm[0]}, {self.max_pwm[0]}] | Saturated Ticks: {self.total_sat_ticks[0]} ({100.0*self.total_sat_ticks[0]/n_ticks:.1f}%)\n")
            self.txt_file.write(f"PWM Roll Range           : [{self.min_pwm[1]}, {self.max_pwm[1]}] | Saturated Ticks: {self.total_sat_ticks[1]} ({100.0*self.total_sat_ticks[1]/n_ticks:.1f}%)\n")
            self.txt_file.write(f"PWM Throttle Range       : [{self.min_pwm[2]}, {self.max_pwm[2]}] | Saturated Ticks: {self.total_sat_ticks[2]} | Floor Enforcements: {self.floor_enforcement_count}\n")
            self.txt_file.write(f"Max Dwell in ±0.8 Box    : {self.max_task4b_box_timer:.2f} s (Task 4B Goal = 10.0s)\n")
            self.txt_file.write(f"Max Dwell in 0.5m Sphere : {self.max_stability_timer:.2f} s\n")
            self.txt_file.write("=================================================================\n")
            self.txt_file.flush()
            self.txt_file.close()
        except Exception:
            pass

        try:
            self.log_file.flush()
            self.log_file.close()
        except Exception:
            pass

        try:
            shutil.copyfile(self.csv_path, self.latest_csv_path)
            shutil.copyfile(self.txt_path, self.latest_txt_path)
            print(f"\n[FLIGHT RECORDER SAVED]")
            print(f"  -> Summary & Events : {self.latest_txt_path}")
            print(f"  -> Full 50Hz CSV    : {self.latest_csv_path}\n")
        except Exception as e:
            print(f"Failed to copy latest logs: {e}")


def out_z_val(val):
    return float(val)


def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico_DampLimited_Steady()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: Disarming and saving flight logs...")
        node.disarm()
    finally:
        node.close_logs()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
