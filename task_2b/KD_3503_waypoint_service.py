#!/usr/bin/env python3
'''
# Team ID:          eYRC#3503
# Theme:            Krishi Drone (KD)
# Author List:      Harsh Patnaik
# Filename:         waypoint_service.py
# Functions:        crop_plant_blocks, segment_plants_watershed, find_infected_plants, 
#                   __init__ (WayPoints), image_callback (WayPoints), 
#                   build_mission_plan (WayPoints), write_mission_log_to_file (WayPoints), 
#                   waypoint_callback (WayPoints), main
# Global variables: None
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from waypoint_navigation.srv import GetWaypoints
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# -------------------------------------------------------------------
# --- Image Processing Functions (All fixes applied) ---
# -------------------------------------------------------------------

def crop_plant_blocks(roi_image, logger):
    '''
    Purpose:
    ---
    Crops the two main plant blocks from the farm's region of interest (ROI).
    It uses Otsu's thresholding and contour filtering based on area to
    identify the four plant beds, then separates them into two blocks.

    Input Arguments:
    ---
    `roi_image` :  [ numpy.ndarray ]
        The image (BGR) of the farm area to be processed.
    
    `logger` :  [ rclpy.logging.Logger ]
        The logger object from the node to print debug messages.

    Returns:
    ---
    `block1` :  [ numpy.ndarray ]
        Cropped image of the first (bottom) block of plant beds.
    
    `block2` :  [ numpy.ndarray ]
        Cropped image of the second (top) block of plant beds.

    Example call:
    ---
    block1, block2 = crop_plant_blocks(farm_roi, self.get_logger())
    '''
    if roi_image is None:
        logger.error("crop_plant_blocks received a None image.")
        raise ValueError("Received a None image for cropping.")
    
    h, w, c = roi_image.shape
    logger.info(f"DEBUG: crop_plant_blocks received image of size (HxW): {h}x{w}")

    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    thresh_val, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.info(f"DEBUG: Otsu's threshold value: {thresh_val}")
    
    cv2.imwrite("debug_crop_gray.jpg", gray_image)
    cv2.imwrite("debug_crop_blurred.jpg", blurred_image)
    cv2.imwrite("debug_crop_thresh.jpg", thresh)
    logger.info("DEBUG: Saved debug_crop_gray.jpg, debug_crop_blurred.jpg, and debug_crop_thresh.jpg")
    
    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"DEBUG: cv2.findContours (using RETR_LIST) found {len(contours)} initial contours.")
    
    # --- Filter contours by area ---
    MIN_BED_AREA = 15000
    MAX_BED_AREA = 35000 
    
    bed_contours = []
    contour_areas = [] 
    for c in contours:
        area = cv2.contourArea(c)
        contour_areas.append(area)
        # Keep only contours that are plausibly a plant bed
        if MIN_BED_AREA < area < MAX_BED_AREA:
            bed_contours.append(c)

    logger.info(f"DEBUG: Top 10 contour areas: {sorted(contour_areas, reverse=True)[:10]}")
    logger.info(f"DEBUG: Found {len(bed_contours)} contours within area range ({MIN_BED_AREA}-{MAX_BED_AREA}).")
    
    if len(bed_contours) < 4:
        logger.error(f"ERROR: Could not find 4 contours in the correct area range. Found {len(bed_contours)}.")
        raise ValueError("Could not detect the 4 plant beds (after area filtering).")
    
    if len(bed_contours) > 4:
        logger.warn(f"Found {len(bed_contours)} beds, using 4 largest.")
        bed_contours = sorted(bed_contours, key=cv2.contourArea, reverse=True)[:4]

    # --- Sort by Y-centroid (pixel Y) to separate top/bottom blocks ---
    centroids = [cv2.moments(c) for c in bed_contours]
    cy = [int(M['m01'] / M['m00']) if M['m00'] != 0 else 0 for M in centroids]
    
    median_y = np.median(cy)
    
    block1_beds = [c for i, c in enumerate(bed_contours) if cy[i] >= median_y] # Bottom half of image
    block2_beds = [c for i, c in enumerate(bed_contours) if cy[i] < median_y]  # Top half of image

    if not block1_beds or not block2_beds:
         logger.error("Failed to separate beds into two blocks. Check Y-sorting.")
         raise ValueError("Could not separate beds into Block 1 and Block 2.")

    # --- Create the cropped images ---
    block1_contour = np.vstack(block1_beds)
    x, y, w, h = cv2.boundingRect(block1_contour)
    block1 = roi_image[y:y+h, x:x+w]
    
    block2_contour = np.vstack(block2_beds)
    x, y, w, h = cv2.boundingRect(block2_contour)
    block2 = roi_image[y:y+h, x:x+w]

    return block1, block2

def segment_plants_watershed(block_image, initial_mask):
    '''
    Purpose:
    ---
    Uses the Watershed algorithm to separate contours of touching objects
    (in this case, plants).

    Input Arguments:
    ---
    `block_image` :  [ numpy.ndarray ]
        The BGR image of the plant block (used by watershed).
    
    `initial_mask` :  [ numpy.ndarray ]
        A binary mask showing all plant matter (healthy + unhealthy).

    Returns:
    ---
    `final_contours` :  [ list ]
        A list of contours, where each contour represents a single,
        separated plant.

    Example call:
    ---
    plant_contours = segment_plants_watershed(block_image, plant_matter_mask)
    '''
    kernel = np.ones((3,3), np.uint8)
    sure_fg = cv2.erode(initial_mask, kernel, iterations=2)
    dilated = cv2.dilate(initial_mask, kernel, iterations=3)
    sure_bg = cv2.threshold(dilated, 1, 128, cv2.THRESH_BINARY_INV)[1]
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(block_image, markers)
    
    final_contours = []
    for label in np.unique(markers):
        if label <= 1: continue
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours.append(contours[0])
        
    return final_contours


def find_infected_plants(block_image, block_id_prefix, infection_threshold, logger):
    '''
    Purpose:
    ---
    Analyzes a cropped plant block to find all infected plants. It uses
    HSV color masking to identify unhealthy pixels and Watershed segmentation
    to separate individual plants.

    Input Arguments:
    ---
    `block_image` :  [ numpy.ndarray ]
        The BGR image of the plant block to analyze.
    
    `block_id_prefix` :  [ str ]
        The prefix for plant IDs in this block (e.g., "P1" or "P2").
    
    `infection_threshold` :  [ int ]
        The minimum number of unhealthy pixels to classify a plant as infected.
    
    `logger` :  [ rclpy.logging.Logger ]
        The logger object from the node to print debug messages.

    Returns:
    ---
    `infected_plant_list` :  [ list ]
        A list of strings of the infected plant IDs (e.g., ["P1C", "P2A"]).

    Example call:
    ---
    infected_list_b1 = find_infected_plants(block1, "P1", 80, self.get_logger())
    '''
    hsv_block = cv2.cvtColor(block_image, cv2.COLOR_BGR2HSV)

    if block_id_prefix == "P1":
        cv2.imwrite("debug_hsv_block_P1.jpg", hsv_block)
    else:
        cv2.imwrite("debug_hsv_block_P2.jpg", hsv_block)

    # Define HSV color ranges for healthy, unhealthy, and flowers
    lower_healthy_green = np.array([30, 40, 40])
    upper_healthy_green = np.array([85, 255, 255])
    lower_unhealthy = np.array([15, 80, 80])
    upper_unhealthy = np.array([35, 255, 255])
    lower_flower = np.array([25, 150, 150])
    upper_flower = np.array([35, 255, 255])

    # Create masks
    healthy_green_mask = cv2.inRange(hsv_block, lower_healthy_green, upper_healthy_green)
    unhealthy_mask = cv2.inRange(hsv_block, lower_unhealthy, upper_unhealthy)
    flower_mask = cv2.inRange(hsv_block, lower_flower, upper_flower)

    # Combine masks to isolate plant matter
    plant_matter_mask = cv2.bitwise_or(healthy_green_mask, unhealthy_mask)
    healthy_parts_mask = cv2.bitwise_or(healthy_green_mask, flower_mask)
    # Isolate only the truly unhealthy pixels (unhealthy minus flowers)
    true_unhealthy_mask = cv2.subtract(plant_matter_mask, healthy_parts_mask)
    
    if block_id_prefix == "P1":
        cv2.imwrite("debug_plant_matter_mask_P1.jpg", plant_matter_mask)
        cv2.imwrite("debug_true_unhealthy_mask_P1.jpg", true_unhealthy_mask)
        logger.info("DEBUG: Saved P1 mask debug images.")
    else:
        cv2.imwrite("debug_plant_matter_mask_P2.jpg", plant_matter_mask)
        cv2.imwrite("debug_true_unhealthy_mask_P2.jpg", true_unhealthy_mask)
        logger.info("DEBUG: Saved P2 mask debug images.")

    # Segment the plants
    plant_contours = segment_plants_watershed(block_image, plant_matter_mask)
    
    if len(plant_contours) < 6:
        logger.warn(f"Block {block_id_prefix}: Watershed found {len(plant_contours)}/6 plants.")
        if not plant_contours: return []
    
    plant_contours = sorted(plant_contours, key=cv2.contourArea, reverse=True)[:6]
    
    # --- Sort plants into 2 rows (by Y-centroid) of 3 (by X-centroid) ---
    centroids = [cv2.moments(c) for c in plant_contours]
    cy = [int(M['m01'] / M['m00']) if M['m00'] != 0 else 0 for M in centroids]
    median_y = np.median(cy)
    
    # Row D/E/F is "top" (lower pixel Y)
    row_DEF = sorted(
        [c for i, c in enumerate(plant_contours) if cy[i] < median_y], 
        key=lambda c: cv2.boundingRect(c)[0]
    )
    # Row A/B/C is "bottom" (higher pixel Y)
    row_ABC = sorted(
        [c for i, c in enumerate(plant_contours) if cy[i] >= median_y], 
        key=lambda c: cv2.boundingRect(c)[0]
    )

    sorted_contours = row_ABC + row_DEF
    plant_ids_suffix = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Fallback sorting logic if rows aren't clearly separated
    if len(sorted_contours) != 6:
         logger.warn("Plant row sorting failed. Using fallback (left/right column sort).")
         cx = [int(M['m10'] / M['m00']) if M['m00'] != 0 else 0 for M in centroids]
         median_x = np.median(cx)
         left_column = sorted([c for i, c in enumerate(plant_contours) if cx[i] < median_x], key=lambda c: cv2.boundingRect(c)[1])
         right_column = sorted([c for i, c in enumerate(plant_contours) if cx[i] >= median_x], key=lambda c: cv2.boundingRect(c)[1])
         sorted_contours = left_column + right_column

    
    infected_plant_list = []
    infection_counts = []

    # Check each of the 6 sorted plants for infection
    for i, contour in enumerate(sorted_contours):
        if i >= 6: break 
        mask = np.zeros(hsv_block.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Count unhealthy pixels within this plant's contour
        infection_in_contour = cv2.bitwise_and(true_unhealthy_mask, true_unhealthy_mask, mask=mask)
        count = cv2.countNonZero(infection_in_contour)
        infection_counts.append(count)

        if count > infection_threshold:
            plant_id = f"{block_id_prefix}{plant_ids_suffix[i]}"
            infected_plant_list.append(plant_id)

    logger.info(f"Block {block_id_prefix} infection counts [A,B,C,D,E,F]: {infection_counts}")
    return infected_plant_list

# -------------------------------------------------------------------
# --- MAIN WayPoints CLASS ---
# -------------------------------------------------------------------

class WayPoints(Node):

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the WayPoints service node. Sets up subscribers, 
        service servers, and loads mission parameters (plant locations, 
        mandatory waypoints).

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        waypoints = WayPoints()
        '''
        super().__init__('waypoints_service')
        
        # self.srv: Service server for 'get_waypoints'
        self.srv = self.create_service(GetWaypoints, 'get_waypoints', self.waypoint_callback)
        
        # self.image_sub: Subscriber to the '/image_raw' topic
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        # self.bridge: OpenCV-ROS Image converter
        self.bridge = CvBridge()
        
        # self.waypoints: The final list of waypoints for the mission
        self.waypoints = [] 
        # self.mission_ready: Flag to indicate if waypoints are generated
        self.mission_ready = False
        # self.processing_done: Flag to ensure image processing runs only once
        self.processing_done = False
        # self.INFECTION_THRESHOLD: Pixel count to classify a plant as infected
        self.INFECTION_THRESHOLD = 80 
        
        # self.mission_log_data: List to store action descriptions for the log file
        self.mission_log_data = []
        
        # self.plant_locations: Dictionary of all possible plant hover/spray coordinates
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
        
        # self.mandatory_points: Dictionary of key navigation points
        self.mandatory_points = {
            'ground_station': [-6.9, 0.09, 32.16],
            'pest_1_hover':   [-9.11, 9.27, 31.27], # 'destination' from Table 2
            'pest_1_pickup':  [-9.07, 9.23, 32.57], # 'pickup' from Table 2
            
            'hh_2_entry':     [-3.26, 8.41, 29.88], # 'hula_hoop_2_entry' from Table 3
            'hh_2_exit':      [0.87, 8.18, 29.05], # 'hula_hoop_2_exit' from Table 3

            'hh_1_entry':     [-3.26, -8.12, 29.48], # 'hula_hoop_1_entry' from Table 8
            'hh_1_exit':      [0.8, -8.04, 29.20],  # 'hula_hoop_1_exit' from Table 8
        }
        
        
        self.get_logger().info('Waypoint Service (Task 2B - All-in-One) is initializing.')
        self.get_logger().warn(f'Waiting for camera image on {self.image_sub.topic_name}...')
        self.get_logger().warn(f'Using infection threshold: {self.INFECTION_THRESHOLD} pixels')

    
    def image_callback(self, msg):
        '''
        Purpose:
        ---
        Callback for the image subscriber. Runs ONCE when an image is received.
        It converts the image, triggers the cropping and infection detection
        functions, and then builds the final mission plan.

        Input Arguments:
        ---
        `msg` :  [ sensor_msgs.msg.Image ]
            The ROS Image message from the camera.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by the /image_raw subscriber)
        '''
        if self.processing_done:
            return
            
        self.get_logger().info('--- Image received! Processing... ---')
        self.processing_done = True 

        try:
            # 1. Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite("debug_received_image.jpg", cv_image)
            
            # 2. Skip ArUco warp. Use the raw image as the Region of Interest.
            roi_image = cv_image
            self.get_logger().info("Skipped ArUco transform. Using raw image as ROI.")

            # --- Crop to the right half (farm area) ---
            h, w, _ = roi_image.shape
            crop_x_start = (w // 2) - 50 
            if crop_x_start < 0: crop_x_start = 0
            farm_roi = roi_image[0:h, crop_x_start:w] 
            self.get_logger().info(f"DEBUG: Cropped full ROI from {w}x{h} to farm-only ROI of {farm_roi.shape[1]}x{farm_roi.shape[0]}")
            cv2.imwrite("debug_farm_roi.jpg", farm_roi)
            # ---

            if farm_roi is None:
                self.get_logger().error("farm_roi is None before calling crop_plant_blocks.")
                return
            h_farm, w_farm, c_farm = farm_roi.shape
            self.get_logger().info(f"DEBUG: Passing image of size (HxW): {h_farm}x{w_farm} to crop_plant_blocks.")

            # 3. Crop blocks from the (unwarped) farm_roi
            block1, block2 = crop_plant_blocks(farm_roi, self.get_logger())
            
            self.get_logger().info("Successfully cropped Block 1 and Block 2.")
            cv2.imwrite("debug_block_1.jpg", block1) # Save final cropped blocks
            cv2.imwrite("debug_block_2.jpg", block2) 

            # 4. Find infected plants in each block
            infected_list_b1 = find_infected_plants(
                block1, "P1", self.INFECTION_THRESHOLD, self.get_logger()
            )
            infected_list_b2 = find_infected_plants(
                block2, "P2", self.INFECTION_THRESHOLD, self.get_logger()
            )
            
            total_infected_list = infected_list_b1 + infected_list_b2
            
            if not total_infected_list:
                self.get_logger().warn("No plants found above infection threshold.")
            
            # 5. Build the final mission plan
            self.build_mission_plan(total_infected_list)

            # 6. Stop subscribing to the camera (Commented out for rqt_graph)
            self.get_logger().info("Shutting down camera subscriber (subscription kept active for rqt_graph).")
            # self.destroy_subscription(self.image_sub) # <-- Kept commented out for submission

        except Exception as e:
            self.get_logger().error(f"Error during plant detection: {e}")
            self.mission_ready = True 


    def build_mission_plan(self, infected_keys):
        '''
        Purpose:
        ---
        Assembles the final `self.waypoints` list using ONLY mandatory
        points. The path is built based on the list of infected plant keys.
        It also populates `self.mission_log_data` for logging.

        Input Arguments:
        ---
        `infected_keys` :  [ list ]
            A list of strings of the infected plant IDs (e.g., ["P1C", "P2A"]).

        Returns:
        ---
        None

        Example call:
        ---
        self.build_mission_plan(['P1C', 'P2A'])
        '''
        self.get_logger().info(f'--- Building MINIMAL mission plan for: {infected_keys} ---')
        
        temp_waypoints = []
        self.mission_log_data = [] # Clear log
        
        try:
            # 1. Start at Ground Station
            temp_waypoints.append(self.mandatory_points['ground_station'])
            self.mission_log_data.append("Ground Station (3s)") # Start/End is important

            # 2. Go to Pesticide Station 1
            temp_waypoints.append(self.mandatory_points['pest_1_hover'])
            self.mission_log_data.append("Pesticide Hover (3s)")
            temp_waypoints.append(self.mandatory_points['pest_1_pickup'])
            self.mission_log_data.append("Pesticide Pickup (3s)")

            # 3. Go to Block 1 (via Hula Hoop 2)
            temp_waypoints.append(self.mandatory_points['hh_2_entry'])
            self.mission_log_data.append("Hula Hoop 2 Entry (0s)")
            temp_waypoints.append(self.mandatory_points['hh_2_exit'])
            self.mission_log_data.append("Hula Hoop 2 Exit (0s)")

            # 4. Visit Infected Plants in Block 1
            block_1_keys = sorted([key for key in infected_keys if key.startswith('P1')])
            self.get_logger().info(f"Visiting Block 1 plants: {block_1_keys}")
            for key in block_1_keys:
                if key in self.plant_locations:
                    temp_waypoints.append(self.plant_locations[key]['hover'])
                    self.mission_log_data.append(f"{key} Hover (3s)")
                    
                    temp_waypoints.append(self.plant_locations[key]['spray'])
                    self.mission_log_data.append(f"{key} Spray (3s)")
                else:
                    self.get_logger().warn(f"Key {key} not found in plant_locations")

            # 5. Visit Infected Plants in Block 2 (Directly from Block 1)
            block_2_keys = sorted([key for key in infected_keys if key.startswith('P2')])
            self.get_logger().info(f"Visiting Block 2 plants: {block_2_keys}")
            for key in block_2_keys:
                if key in self.plant_locations:
                    temp_waypoints.append(self.plant_locations[key]['hover'])
                    self.mission_log_data.append(f"{key} Hover (3s)")
                    
                    temp_waypoints.append(self.plant_locations[key]['spray'])
                    self.mission_log_data.append(f"{key} Spray (3s)")
                else:
                    self.get_logger().warn(f"Key {key} not found in plant_locations")

            # 6. Return to Ground (via Hula Hoop 1) - REVERSED ORDER
            temp_waypoints.append(self.mandatory_points['hh_1_exit'])   # Go to exit first
            self.mission_log_data.append("Hula Hoop 1 Exit (0s)")
            temp_waypoints.append(self.mandatory_points['hh_1_entry']) # Go to entry second
            self.mission_log_data.append("Hula Hoop 1 Entry (0s)")

            # 7. Final waypoint: Ground Station
            temp_waypoints.append(self.mandatory_points['ground_station'])
            self.mission_log_data.append("Ground Station (3s)")

            # --- Finalize Mission ---
            self.waypoints = temp_waypoints
            
            # --- NEW: Write the log file ---
            self.write_mission_log_to_file()
            
            self.get_logger().info(
                f'--- MINIMAL MISSION PLAN BUILT. Total waypoints: {len(self.waypoints)}. Service is READY. ---'
            )
        
        except Exception as e:
            self.get_logger().error(f"Error in build_mission_plan: {e}")
        
        self.mission_ready = True

    def write_mission_log_to_file(self):
        '''
        Purpose:
        ---
        Writes the generated mission plan (coordinates and actions)
        to a text file ("task2b_mission_log.txt") for easy debugging.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.write_mission_log_to_file()
        '''
        log_filename = "task2b_mission_log.txt"
        try:
            with open(log_filename, 'w') as f:
                f.write("--- Task 2B Minimal Mission Plan ---\n")
                f.write(f"Total Waypoints: {len(self.waypoints)}\n\n")
                
                # Create header
                f.write(f"{'#':<3} | {'Coordinates':<25} | {'Action (Required Dwell)':<25}\n")
                f.write("-" * 57 + "\n")
                
                # Write each waypoint and its corresponding action
                for i, (wp, action_str) in enumerate(zip(self.waypoints, self.mission_log_data)):
                    coord_str = f"[{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]"
                    f.write(f"{i+1:<3} | {coord_str:<25} | {action_str:<25}\n")
                    
            self.get_logger().info(f"--- Mission plan successfully written to {log_filename} ---")
            
        except Exception as e:
            self.get_logger().error(f"Failed to write mission log file: {e}")


    def waypoint_callback(self, request, response):
        '''
        Purpose:
        ---
        Callback function executed when the 'get_waypoints' service is called.
        It checks if the mission plan is ready before responding with
        the list of waypoints.

        Input Arguments:
        ---
        `request` :  [ GetWaypoints.Request ]
            The service request object (empty for this service).
        
        `response` :  [ GetWaypoints.Response ]
            The service response object to be populated with waypoints.

        Returns:
        ---
        `response` :  [ GetWaypoints.Response ]
            The populated service response.

        Example call:
        ---
        (Called automatically by the service server)
        '''
        
        if not self.mission_ready:
            self.get_logger().warn("Service called, but mission plan is not ready. (Waiting for camera image...)")
            response.waypoints = [] # Return an empty list
            return response

        self.get_logger().info(f"--- Incoming request... Sending {len(self.waypoints)} waypoints. ---")
        response.waypoints = [] 

        for wp in self.waypoints:
            point = Point()
            point.x = wp[0]
            point.y = wp[1]
            point.z = wp[2]
            response.waypoints.append(point) 

        self.get_logger().info(f"Sending response with {len(response.waypoints)} waypoints.")
        return response

def main(args=None):
    '''
    Purpose:
    ---
    Initializes the ROS 2 system and the WayPoints node.
    It spins the node, allowing it to provide services and
    respond to subscribers.

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
    waypoints = WayPoints()
    try:
        rclpy.spin(waypoints)
    except KeyboardInterrupt:
        waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoints.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()