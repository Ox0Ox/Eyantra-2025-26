#!/usr/bin/env python3
'''
# Team ID:          eYRC#3503
# Theme:            Krishi Drone (KD)
# Author List:      Harsh Patnaik
# Filename:         KD_3503_waypoint_service.py
# Functions:        __init__, waypoint_callback, main
# Global variables: None
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
# Import the custom service
from waypoint_navigation.srv import GetWaypoints

class WayPoints(Node):

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the WayPoints service node. It defines the hardcoded
        list of waypoints and creates the 'get_waypoints' service server.

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
        # Service name is 'get_waypoints'
        self.srv = self.create_service(GetWaypoints, 'get_waypoints', self.waypoint_callback)
        
        # self.waypoints: List of [x, y, z] coordinates for the drone's mission
        self.waypoints = [
            [-7.00, 0.00, 29.22],
            [-7.64, 3.06, 29.22],
            [-8.22, 6.02, 29.22],
            [-9.11, 9.27, 29.27],
            [-5.98, 8.81, 29.27],
            [-3.26, 8.41, 29.88],
            [0.87, 8.18, 29.05],
            [3.93, 7.35, 29.05]
        ]
        self.get_logger().info('Waypoint Service is ready and waiting for a request.')

    
    def waypoint_callback(self, request, response):
        '''
        Purpose:
        ---
        Callback function executed when the 'get_waypoints' service is called. 
        It populates the service response with the predefined list of waypoints.

        Input Arguments:
        ---
        `request` : [ waypoint_navigation.srv.GetWaypoints.Request ]
            The service request object (unused in this implementation).
        `response` : [ waypoint_navigation.srv.GetWaypoints.Response ]
            The service response object to be filled with waypoints.

        Returns:
        ---
        `response` : [ waypoint_navigation.srv.GetWaypoints.Response ]
            The service response object containing the list of waypoints.

        Example call:
        ---
        (Called automatically by ROS 2 Service Server)
        '''

        self.get_logger().info("--- Incoming request for Waypoints... ---")

        # Initialize the waypoints list directly on the response object
        response.waypoints = [] 

        # Populate the response with Point messages containing waypoint coordinates
        for wp in self.waypoints:
            point = Point() # Use Point, as defined in the .srv file
            point.x = wp[0]
            point.y = wp[1]
            point.z = wp[2]
            # Append the Point message directly to the response list
            response.waypoints.append(point) 

        self.get_logger().info(f"Sending response with {len(response.waypoints)} waypoints.")
        return response

def main():
    '''
    Purpose:
    ---
    Initializes the ROS 2 system, creates an instance of the WayPoints
    service node, and spins the node to keep it alive and responsive
    to service requests.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    if __name__ == '__main__':
        main()
    '''
    rclpy.init()
    waypoints = WayPoints()

    try:
        # Spin the node to process service calls
        rclpy.spin(waypoints)
    except KeyboardInterrupt:
        waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        # Cleanly destroy the node and shutdown rclpy
        waypoints.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
