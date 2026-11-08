#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import json

class InfectedPlantsDetector(Node):
    def __init__(self):
        super().__init__('infected_plants_detector')
        
        # ROS2 Setup
        self.subscription = self.create_subscription(
            Image, 
            '/image_raw', 
            self.image_callback, 
            10
        )
        self.publisher_ = self.create_publisher(String, '/detected_plants', 10)
        self.bridge = CvBridge()
        
        # Configuration parameters (from test.py)
        self.ROI_SIZE = 1000
        self.WHITE_BED_THRESHOLD = 180
        self.INFECTION_PIXEL_THRESHOLD = 600
        self.INFECTION_DENSITY_THRESHOLD = 0.12
        self.MINIMUM_INFECTION_PIXELS = 150
        self.VALID_IDS = [80, 85, 90, 95]
        
        # Frame processing control
        self.frame_count = 0
        self.process_every_n_frames = 5  # Process every 5th frame to reduce load
        
        self.get_logger().info('Infected Plants Detector Node Initialized')
        self.get_logger().info(f'Detection Thresholds: {self.INFECTION_PIXEL_THRESHOLD}px OR '
                              f'{self.INFECTION_DENSITY_THRESHOLD*100:.0f}% density '
                              f'(min {self.MINIMUM_INFECTION_PIXELS}px)')

    def detect_and_transform(self, image):
        """Detect ArUco markers and perform perspective transform - FROM TEST.PY"""
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, rejected = detector.detectMarkers(image)

            # Debug visualization
            marker_viz = image.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(marker_viz, corners, ids)
            cv2.imwrite("ros_step1_detected_markers.jpg", marker_viz)

            if ids is None or len(ids) < 4:
                self.get_logger().warn(
                    f'Insufficient markers: found {len(ids) if ids is not None else 0}, need 4'
                )
                return None, None

            # Build marker ID map
            id_map = {}
            for i in range(len(ids)):
                marker_id = ids[i][0]
                marker_corners = corners[i][0]
                center = np.mean(marker_corners, axis=0)
                id_map[marker_id] = center

            if not all(rid in id_map for rid in self.VALID_IDS):
                self.get_logger().warn(
                    f'Missing required markers. Found: {list(id_map.keys())}, Need: {self.VALID_IDS}'
                )
                return None, None

            # Perspective transform
            src_points = np.float32([
                id_map[85],  # Top Left
                id_map[90],  # Top Right
                id_map[95],  # Bottom Right
                id_map[80]   # Bottom Left
            ])
            
            dst_points = np.float32([
                [0, 0],
                [self.ROI_SIZE-1, 0],
                [self.ROI_SIZE-1, self.ROI_SIZE-1],
                [0, self.ROI_SIZE-1]
            ])

            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(image, matrix, (self.ROI_SIZE, self.ROI_SIZE))
            
            cv2.imwrite("ros_step2_warped_arena.jpg", warped)
            
            marker_ids_sorted = sorted([int(id_val) for id_val in id_map.keys()])
            return warped, marker_ids_sorted
            
        except Exception as e:
            self.get_logger().error(f'ArUco detection failed: {str(e)}')
            return None, None

    def segment_plants_watershed(self, block_image, initial_mask, prefix):
        """Segment individual plants using watershed algorithm - FROM TEST.PY"""
        try:
            if cv2.countNonZero(initial_mask) == 0:
                return []
            
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_large = np.ones((5, 5), np.uint8)
            
            # Close small gaps
            cleaned_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
            
            # Erode to separate touching plants
            eroded = cv2.erode(cleaned_mask, kernel_small, iterations=1)
            
            # Dilate back to restore size
            restored = cv2.dilate(eroded, kernel_small, iterations=1)
            
            # Watershed preparation
            sure_bg = cv2.dilate(restored, kernel_large, iterations=3)
            dist_transform = cv2.distanceTransform(restored, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            cv2.imwrite(f"ros_step8a_{prefix}_watershed_seeds.jpg", sure_fg)
            
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(block_image, markers)
            
            marker_viz = np.uint8(np.clip(markers * 30, 0, 255))
            cv2.imwrite(f"ros_step8b_{prefix}_watershed_blobs.jpg", marker_viz)
            
            # Extract individual plant contours
            final_contours = []
            for label in np.unique(markers):
                if label <= 1:
                    continue
                m_mask = np.zeros(markers.shape, dtype="uint8")
                m_mask[markers == label] = 255
                cnts, _ = cv2.findContours(m_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    contour = cnts[0]
                    if cv2.contourArea(contour) > 500:
                        final_contours.append(contour)
            
            return final_contours
            
        except Exception as e:
            self.get_logger().error(f'Watershed segmentation failed for {prefix}: {str(e)}')
            return []

    def find_most_infected_plant(self, block_image, block_mask, prefix, debug_canvas, offset_x, offset_y):
        """
        Detect plants using TEST.PY watershed segmentation
        Identify infection using DENSITY-BASED logic - FROM TEST.PY
        """
        try:
            # Apply the PERFECT MASK
            masked_block = cv2.bitwise_and(block_image, block_image, mask=block_mask)
            hsv = cv2.cvtColor(masked_block, cv2.COLOR_BGR2HSV)
            
            # --- COLOR THRESHOLDS FROM TEST.PY ---
            # Healthy green
            lower_healthy_green = np.array([30, 10, 25])
            upper_healthy_green = np.array([85, 255, 255])
            
            # Unhealthy: Brown/Red/Yellow
            # Mask 1: Standard Orange-Red (0-40)
            lower1 = np.array([0, 30, 30])
            upper1 = np.array([40, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)

            # Mask 2: Magenta-Red (165-180)
            lower2 = np.array([165, 30, 30])
            upper2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower2, upper2)

            # Floor/stem yellows
            lower_flower = np.array([25, 150, 150])
            upper_flower = np.array([35, 255, 255])
            
            healthy_green_mask = cv2.inRange(hsv, lower_healthy_green, upper_healthy_green)
            unhealthy_mask = cv2.bitwise_or(mask1, mask2)
            flower_mask = cv2.inRange(hsv, lower_flower, upper_flower)
            
            # True unhealthy = unhealthy MINUS healthy MINUS floor
            true_unhealthy_mask = cv2.subtract(unhealthy_mask, healthy_green_mask)
            true_unhealthy_mask = cv2.subtract(true_unhealthy_mask, flower_mask)
            
            # Densify sparse infections
            kernel_close = np.ones((3, 3), np.uint8)
            true_unhealthy_mask = cv2.morphologyEx(
                true_unhealthy_mask, 
                cv2.MORPH_CLOSE, 
                kernel_close, 
                iterations=1
            )
            
            cv2.imwrite(f"ros_step6_{prefix}_infection_mask.jpg", true_unhealthy_mask)
            
            # Combined plant matter for segmentation
            plant_matter = cv2.bitwise_or(healthy_green_mask, unhealthy_mask)
            cv2.imwrite(f"ros_step7_{prefix}_plant_matter.jpg", plant_matter)
            
            # TEST.PY SEGMENTATION: Use watershed
            plant_contours = self.segment_plants_watershed(block_image, plant_matter, prefix)
            
            # Filter noise < 5% of largest
            if plant_contours:
                largest_area = max(cv2.contourArea(c) for c in plant_contours)
                plant_contours = [c for c in plant_contours if cv2.contourArea(c) > (largest_area * 0.05)]
            
            # Keep top 6 largest
            plant_contours = sorted(plant_contours, key=cv2.contourArea, reverse=True)[:6]
            
            if not plant_contours:
                self.get_logger().warn(f'No plants detected in {prefix}')
                return "None"
            
            # Grid Sort (A-C Bottom, D-F Top)
            centroids = [cv2.moments(c) for c in plant_contours]
            cy = [int(M['m01'] / M['m00']) if M['m00'] != 0 else 0 for M in centroids]
            median_y = np.median(cy)
            
            # Bottom row (ABC) - higher Y values
            row_ABC = sorted(
                [c for i, c in enumerate(plant_contours) if cy[i] >= median_y], 
                key=lambda c: cv2.boundingRect(c)[0]
            )
            # Top row (DEF) - lower Y values
            row_DEF = sorted(
                [c for i, c in enumerate(plant_contours) if cy[i] < median_y], 
                key=lambda c: cv2.boundingRect(c)[0]
            )
            
            sorted_contours = row_ABC + row_DEF
            plant_ids = ['A', 'B', 'C', 'D', 'E', 'F']
            max_px, winner = 0, "None"
            
            self.get_logger().info(f'{prefix} Plant Diagnosis:')
            
            # Analyze each plant
            for i, contour in enumerate(sorted_contours):
                if i >= 6:
                    break
                
                p_id = f"{prefix}{plant_ids[i]}"
                
                # Create mask for this plant
                m = np.zeros(hsv.shape[:2], dtype="uint8")
                cv2.drawContours(m, [contour], -1, 255, -1)
                
                # Get plant area
                plant_area = cv2.countNonZero(m)
                
                # Count infected pixels within plant
                infection_in_contour = cv2.bitwise_and(true_unhealthy_mask, true_unhealthy_mask, mask=m)
                px_count = cv2.countNonZero(infection_in_contour)
                
                # Calculate infection density
                infection_density = px_count / plant_area if plant_area > 0 else 0
                
                # DENSITY-BASED DETECTION LOGIC
                meets_absolute = px_count >= self.INFECTION_PIXEL_THRESHOLD
                meets_density = (px_count >= self.MINIMUM_INFECTION_PIXELS) and \
                               (infection_density >= self.INFECTION_DENSITY_THRESHOLD)
                
                is_infected = meets_absolute or meets_density
                
                bx, by, bw, bh = cv2.boundingRect(contour)
                
                # Determine status string
                if is_infected:
                    if meets_absolute:
                        status_str = f"INFECTED [ABSOLUTE: {px_count}px]"
                    else:
                        status_str = f"INFECTED [DENSITY: {infection_density:.1%}]"
                else:
                    status_str = f"Healthy"
                
                self.get_logger().info(
                    f'  {p_id}: {status_str} (Pixels: {px_count}/{plant_area} = {infection_density:.1%})'
                )
                
                # Color coding
                color = (0, 0, 255) if is_infected else (0, 255, 0)
                
                # Draw on debug canvas
                cv2.rectangle(
                    debug_canvas, 
                    (bx + offset_x, by + offset_y), 
                    (bx + bw + offset_x, by + bh + offset_y), 
                    color, 2
                )
                
                # Label
                label_text = f"{p_id}:{px_count}px|{infection_density:.0%}"
                cv2.putText(
                    debug_canvas, 
                    label_text, 
                    (bx + offset_x, by + offset_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
                )
                
                # Track most infected plant
                if is_infected and px_count > max_px:
                    max_px, winner = px_count, p_id
            
            return winner
            
        except Exception as e:
            self.get_logger().error(f'Plant analysis failed for {prefix}: {str(e)}')
            return "None"

    def process_assembled_block(self, warped, bed_list, block_name, debug_canvas):
        """Process a block using TEST.PY's perfect mask approach"""
        try:
            if not bed_list:
                return "None"
            
            # Create PERFECT MASK
            combined = np.vstack(bed_list)
            x, y, w, h = cv2.boundingRect(combined)
            
            # Empty mask - only draw valid bed rectangles
            block_mask = np.zeros((h, w), dtype=np.uint8)
            
            for rect_contour in bed_list:
                rect_shifted = rect_contour - [x, y]
                cv2.drawContours(block_mask, [rect_shifted], -1, 255, -1)
            
            cv2.imwrite(f"ros_step5b_{block_name}_clean_mask.jpg", block_mask)
            
            block_region = warped[y:y+h, x:x+w]
            
            return self.find_most_infected_plant(
                block_region, 
                block_mask, 
                block_name, 
                debug_canvas, 
                x, 
                y
            )
            
        except Exception as e:
            self.get_logger().error(f'Block processing failed for {block_name}: {str(e)}')
            return "None"

    def image_callback(self, msg):
        """ROS2 image callback with integrated test.py logic"""
        try:
            # Frame rate control
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frames != 0:
                return
            
            # Convert ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if frame is None or frame.size == 0:
                self.get_logger().error('Received empty frame')
                return
            
            self.get_logger().info('Processing new frame...')
            
            # Step 1-2: Detect markers and transform
            warped, marker_ids = self.detect_and_transform(frame)
            if warped is None:
                self.get_logger().warn('Failed to transform image - skipping frame')
                return
            
            debug_canvas = warped.copy()
            
            # Step 3: Create bed mask using thresholding
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, self.WHITE_BED_THRESHOLD, 255, cv2.THRESH_BINARY)
            cv2.imwrite("ros_step3_bed_binary_mask.jpg", thresh)
            
            # Step 4: Find bed segment contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bed_segments = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if 8000 < area < 45000 and x > 450 and w > h:
                    bed_segments.append(c)
            
            bed_viz = warped.copy()
            cv2.drawContours(bed_viz, bed_segments, -1, (255, 0, 255), 2)
            cv2.imwrite("ros_step4_detected_bed_segments.jpg", bed_viz)
            
            if len(bed_segments) < 2:
                self.get_logger().warn(
                    f'Insufficient bed segments: found {len(bed_segments)}, need at least 2'
                )
                return
            
            # Step 5: Group rectangles into P1 (bottom) and P2 (top)
            centroids = [cv2.moments(c) for c in bed_segments]
            cy = [int(M['m01'] / M['m00']) if M['m00'] != 0 else 0 for M in centroids]
            median_y = np.median(cy)
            
            block_P1 = [c for i, c in enumerate(bed_segments) if cy[i] >= median_y]
            block_P2 = [c for i, c in enumerate(bed_segments) if cy[i] < median_y]
            
            block_sort_viz = warped.copy()
            cv2.drawContours(block_sort_viz, block_P1, -1, (0, 0, 255), 3)
            cv2.drawContours(block_sort_viz, block_P2, -1, (255, 0, 0), 3)
            cv2.imwrite("ros_step5_block_sorting_result.jpg", block_sort_viz)
            
            # Step 6-8: Process each block
            result_P1 = self.process_assembled_block(warped, block_P1, "P1", debug_canvas)
            result_P2 = self.process_assembled_block(warped, block_P2, "P2", debug_canvas)
            
            # Step 9: Save final diagnosed image
            cv2.imwrite("ros_step9_final_diagnosed_labels.jpg", debug_canvas)
            
            # Prepare results - MATCH BOILERPLATE FORMAT
            # Always publish both P1 and P2, use placeholder if no infection found
            infected_plants = []
            
            # Add P1 result (or placeholder)
            if result_P1 != "None":
                infected_plants.append(result_P1)
            else:
                infected_plants.append("P1X")  # Placeholder when no infection in P1
            
            # Add P2 result (or placeholder)
            if result_P2 != "None":
                infected_plants.append(result_P2)
            else:
                infected_plants.append("P2X")  # Placeholder when no infection in P2
            
            # Publish results in boilerplate format
            msg = String()
            msg.data = json.dumps(infected_plants)  # Convert list to string
            self.publisher_.publish(msg)
            
            self.get_logger().info(f'Detection complete - Published: {infected_plants}')
            self.get_logger().info(f'Marker IDs: {marker_ids}')
            self.get_logger().info(f'Block P1 (bottom): {result_P1}')
            self.get_logger().info(f'Block P2 (top): {result_P2}')
            
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge conversion failed: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Critical error in image processing: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    try:
        rclpy.init(args=args)
        node = InfectedPlantsDetector()
        
        print("\n" + "=" * 60)
        print("ROS2 INFECTED PLANTS DETECTOR NODE - STARTED")
        print("=" * 60)
        print("Waiting for images on /image_raw topic...")
        print("Publishing results to /detected_plants topic")
        print("=" * 60 + "\n")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        print("Node terminated")

if __name__ == '__main__':
    main()