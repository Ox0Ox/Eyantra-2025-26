#!/usr/bin/env python3
'''
# Team ID:          eYRC#3503
# Theme:            Krishi Drone (KD)
# Author List:      Harsh Patnaik
# Filename:         KD_3503_pico_client.py
# Functions:        __init__, send_goal, goal_response_callback, get_result_callback,
#                   feedback_callback, send_request, receive_goals, main
# Global variables: None
'''

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import asyncio
import threading
from rclpy.executors import MultiThreadedExecutor

# Import the action and service definitions
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints
from geometry_msgs.msg import Point 

# Import GoalStatus for result checking
from action_msgs.msg import GoalStatus

class WayPointClient(Node):

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the WayPointClient node. Creates the action client
        for 'waypoint_navigation' and the service client for 'get_waypoints'.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        waypoint_client = WayPointClient()
        '''
        super().__init__('waypoint_client')
        self.get_logger().info('Waypoint Client node started.')
        
        # self.goals: List to store waypoints [x,y,z] received from the service
        self.goals = []
        
        # self.goal_index: Index of the current waypoint being processed
        self.goal_index = 0

        # --- ADDED FOR SIMULATED TIME TRACKING ---
        # self.mission_start_time: Stores the sim time when the first goal is sent
        self.mission_start_time = None 
        # --- END ---
        
        # self.action_client: Client for the NavToWaypoint action server
        self.action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')
        
        # self.cli: Client for the GetWaypoints service
        self.cli = self.create_client(GetWaypoints, 'get_waypoints')
        
        self.get_logger().info('Waiting for "get_waypoints" service...')
        
        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('--- "get_waypoints" Service Found! ---')
        
        # self.req: Request object for the GetWaypoints service
        self.req = GetWaypoints.Request()
        
    def send_goal(self, waypoint):
        '''
        Purpose:
        ---
        Constructs a NavToWaypoint.Goal message and sends it to the action
        server. Registers callbacks for goal response, feedback, and result.

        Input Arguments:
        ---
        `waypoint` :  [ list ]
            A list containing the [x, y, z] coordinates for the goal.

        Returns:
        ---
        None

        Example call:
        ---
        self.send_goal([-7.0, 0.0, 29.22])
        '''

        # Create a NavToWaypoint goal object.
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.x = waypoint[0] 
        goal_msg.waypoint.y = waypoint[1]
        goal_msg.waypoint.z = waypoint[2]

        self.get_logger().info(f'--- Sending goal {self.goal_index + 1}: {waypoint} ---')
        
        # Send the goal asynchronously
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )    
        
        # Add a callback for when the goal is accepted or rejected
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        '''
        Purpose:
        ---
        Callback triggered when the server responds to the goal request
        (accepts or rejects). If accepted, it requests the final result.

        Input Arguments:
        ---
        `future` :  [ rclpy.Future ]
            A future object containing the goal handle.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by self.send_goal_future)
        '''
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('!!! GOAL REJECTED by server!!!')
            rclpy.shutdown() # Stop mission on failure
            return

        self.get_logger().info('--- Goal ACCEPTED by server :) ---')
        
        # Request the result, register callback for when it's done
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        '''
        Purpose:
        ---
        Callback triggered when the action goal is complete (Succeeded,
        Aborted, or Canceled). It logs the result, and if successful,
        advances to the next goal or shuts down if the mission is complete.

        Input Arguments:
        ---
        `future` :  [ rclpy.Future ]
            A future object containing the final result of the action.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by self.get_result_future)
        '''
        result_handle = future.result() # Get the result handle
        
        status = result_handle.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            result = result_handle.result # Access the actual result message
            self.get_logger().info(
                f'--- RESULT: Goal {self.goal_index + 1} Succeeded. '
                f'Time: {result.time_taken:.2f}s ---'
            ) 
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('!!! Goal ABORTED by the server. !!!')
            rclpy.shutdown() # Stop mission on failure
            return
        elif status == GoalStatus.STATUS_CANCELED:
             self.get_logger().info('Goal was CANCELED.')
             rclpy.shutdown() # Stop mission on failure
             return
        else:
             self.get_logger().warn(f'Goal finished with unknown status: {status}')
             rclpy.shutdown() # Stop mission on failure
             return

        # --- Only proceed if SUCCEEDED ---
        self.goal_index += 1

        # Check if there are more waypoints
        if self.goal_index < len(self.goals):
            # Send the next goal
            self.get_logger().info(f'Proceeding to goal {self.goal_index + 1}...')
            self.send_goal(self.goals[self.goal_index])
        else:
            # --- All goals completed ---

            # --- ADDED FOR SIMULATED TIME TRACKING ---
            if self.mission_start_time is not None:
                # Get the final simulated time
                mission_end_time = self.get_clock().now()
                # Calculate duration in nanoseconds
                duration_ns = mission_end_time.nanoseconds - self.mission_start_time.nanoseconds
                # Convert to seconds
                duration_sec = duration_ns / 1e9
                self.get_logger().warn(
                    f'--- TOTAL MISSION TIME: {duration_sec:.2f} simulated seconds ---'
                )
            # --- END ---

            self.get_logger().info('--- MISSION COMPLETE: All waypoints reached successfully! ---')
            rclpy.shutdown() # Shutdown after last goal

    def feedback_callback(self, feedback_msg):
        '''
        Purpose:
        ---
        Callback triggered when the action server sends periodic feedback
        (e.g., current position, stabilization time).

        Input Arguments:
        ---
        `feedback_msg` :  [ NavToWaypoint.Feedback ]
            The feedback message from the action server.

        Returns:
        ---
        None

        Example call:
        ---
        (Called automatically by the Action Client)
        '''
        feedback = feedback_msg.feedback
        x = feedback.current_pose.pose.position.x
        y = feedback.current_pose.pose.position.y
        z = feedback.current_pose.pose.position.z
        
        # The server populates 'sec' with the stabilization time (a hack)
        t = feedback.current_pose.header.stamp.sec 
        
        self.get_logger().info(
            f'Feedback: Pos:[{x:.2f}, {y:.2f}, {z:.2f}], Stabilized for: {t}s',
            throttle_duration_sec=1.0 # Throttle logging
        )

    def send_request(self):
        '''
        Purpose:
        ---
        Calls the 'get_waypoints' service asynchronously.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `future` :  [ rclpy.Future ]
            A future object for the service call.

        Example call:
        ---
        self.future = self.send_request()
        '''
        # The request object has no fields
        self.get_logger().info('Requesting waypoints from service...')
        return self.cli.call_async(self.req)
    
    async def receive_goals(self):
        '''
        Purpose:
        ---
        Asynchronously requests waypoints from the service. Once received,
        it populates `self.goals`, waits for the action server, and
        sends the *first* goal to start the mission.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        asyncio.run(waypoint_client.receive_goals())
        '''
        self.future = self.send_request()
        
        # Asynchronously wait for the service response
        response = await self.future 
        
        if response is None:
            self.get_logger().error('!!! Service call failed (no response). Shutting down. !!!')
            rclpy.shutdown()
            return

        self.get_logger().info('--- Waypoints received successfully. ---')

        # Correctly iterate over the list of Point messages in the response
        for point in response.waypoints: 
            # Extract coordinates directly from the Point message
            waypoint = [point.x, point.y, point.z] 
            self.goals.append(waypoint)
            self.get_logger().info(f'Waypoint Added: {waypoint}')

        if not self.goals:
            self.get_logger().warn('Received empty waypoint list. Shutting down.')
            rclpy.shutdown()
            return

        # Wait for the action server before sending the first goal
        self.get_logger().info('Waiting for "waypoint_navigation" action server...')
        if not self.action_client.wait_for_server(timeout_sec=5.0):
             self.get_logger().error('!!! Action server not available after 5s. Shutting down. !!!')
             rclpy.shutdown()
             return
             
        self.get_logger().info('--- Action Server Found! ---')
        
        # --- ADDED FOR SIMULATED TIME TRACKING ---
        # Record the start time *just before* sending the first goal
        self.mission_start_time = self.get_clock().now()
        self.get_logger().warn(
            f'--- MISSION STARTED at simulated time: {self.mission_start_time.nanoseconds / 1e9:.2f}s ---'
        )
        # --- END ---

        # Send the first goal
        self.send_goal(self.goals[0])
    

def main(args=None):
    '''
    Purpose:
    ---
    Initializes the ROS 2 system and the WayPointClient. It uses a
    MultiThreadedExecutor in a separate thread to handle all ROS 2
    callbacks (service and action). It then runs the async `receive_goals`
    function to start the mission.

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
    waypoint_client = None
    executor = None
    try:
        waypoint_client = WayPointClient()
        
        # We need an executor to spin the node for callbacks
        executor = MultiThreadedExecutor()
        executor.add_node(waypoint_client)

        # Run the executor in a separate thread
        # This thread will handle all action callbacks and service responses
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        
        waypoint_client.get_logger().info("Starting executor thread...")
        executor_thread.start()
        
        # Now that the node is spinning, run the async setup function
        # This blocks the *main* thread until receive_goals() returns
        waypoint_client.get_logger().info("Running async goal receiver...")
        asyncio.run(waypoint_client.receive_goals())
        
        # After receive_goals() sends the first goal, the callbacks
        # (goal_response, get_result) will handle the rest.
        # We just need to keep the main thread alive until rclpy.shutdown()
        # is called from one of those callbacks.
        
        waypoint_client.get_logger().info("Setup complete. Client is running and waiting for action results...")
        
        # Keep main thread alive while rclpy is running
        while rclpy.ok():
            time.sleep(0.1)

    except KeyboardInterrupt:
        if waypoint_client:
            waypoint_client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    except Exception as e:
        if waypoint_client:
            waypoint_client.get_logger().error(f"Unhandled exception in main: {e}")
    finally:
        # Stop the executor and shut down
        if executor:
            executor.shutdown()
        if waypoint_client and rclpy.ok():
            waypoint_client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Wait for the spin thread to exit
        if 'executor_thread' in locals() and executor_thread.is_alive():
            executor_thread.join()

if __name__ == '__main__':
    main()
