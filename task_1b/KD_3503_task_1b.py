#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.srv import SetPen, TeleportAbsolute
import math
import time

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        # Create a publisher for velocity commands
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Create service clients
        self.set_pen_client = self.create_client(SetPen, '/turtle1/set_pen')
        self.teleport_client = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')

        # Wait for services to be available
        while not self.set_pen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SetPen service not available, waiting again...')
        while not self.teleport_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('TeleportAbsolute service not available, waiting again...')
            
        self.get_logger().info('Services are ready. Starting to draw.')
        self.draw_drone()

    def call_set_pen_service(self, r, g, b, width, off):
        """Helper function to call the SetPen service."""
        request = SetPen.Request()
        request.r = r
        request.g = g
        request.b = b
        request.width = width
        request.off = off
        self.set_pen_client.call_async(request)
        time.sleep(0.1) # Give a moment for the service call to process

    def call_teleport_service(self, x, y, theta):
        """Helper function to call the TeleportAbsolute service."""
        request = TeleportAbsolute.Request()
        request.x = x
        request.y = y
        request.theta = theta
        self.teleport_client.call_async(request)
        time.sleep(0.1) # Give a moment for the service call to process

    def draw_circle(self, center_x, center_y, radius):
        """Draws a circle using cmd_vel publisher."""
        self.get_logger().info(f'Drawing circle at ({center_x}, {center_y})')
        # Lift pen, go to the starting point of the circle, then lower pen
        self.call_set_pen_service(0, 0, 0, 2, 1) # Pen off
        self.call_teleport_service(center_x + radius, center_y, math.pi / 2)
        self.call_set_pen_service(0, 0, 0, 2, 0) # Pen on

        # Set velocities to draw a circle
        twist = Twist()
        twist.linear.x = 2.0 * math.pi * radius / 10.0 # Speed to complete circle in 10s
        twist.angular.z = 2.0 * math.pi / 10.0 # Angular velocity

        # Publish for the duration of one circle
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < 10.0:
            self.publisher.publish(twist)
            time.sleep(0.05)
        
        # Stop the turtle
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
    
    def draw_line_segment(self, x1, y1, x2, y2):
        """Draws a line segment by teleporting."""
        self.call_set_pen_service(0, 0, 0, 2, 1) # Pen off
        self.call_teleport_service(x1, y1, 0.0)
        self.call_set_pen_service(0, 0, 0, 2, 0) # Pen on
        self.call_teleport_service(x2, y2, 0.0)

    def draw_drone(self):
        """Main function to orchestrate the drawing."""
        self.get_logger().info('--- EXECUTING THE FINAL CORRECTED SCRIPT ---') # <-- ADD THIS 
        # Propellers (Circles)
        propeller_centers = [(2.0, 2.0), (2.0, 8.0), (8.0, 8.0), (8.0, 2.0)]
        propeller_radius = 1.0 # Diameter is 2.0
        for x, y in propeller_centers:
            self.draw_circle(x, y, propeller_radius)

        # Drone Frame (Diamond)
        self.get_logger().info('Drawing drone frame.')
        diamond_vertices = [(3.0, 5.0), (5.0, 7.0), (7.0, 5.0), (5.0, 3.0)]
        self.draw_line_segment(diamond_vertices[0][0], diamond_vertices[0][1], diamond_vertices[1][0], diamond_vertices[1][1])
        self.draw_line_segment(diamond_vertices[1][0], diamond_vertices[1][1], diamond_vertices[2][0], diamond_vertices[2][1])
        self.draw_line_segment(diamond_vertices[2][0], diamond_vertices[2][1], diamond_vertices[3][0], diamond_vertices[3][1])
        self.draw_line_segment(diamond_vertices[3][0], diamond_vertices[3][1], diamond_vertices[0][0], diamond_vertices[0][1])

        # Connecting Arms
        self.get_logger().info('Drawing connecting arms.')
        # Define propeller centers and diamond midpoints
        p_bottom_left  = (2.0, 2.0)
        p_top_left     = (2.0, 8.0)
        p_top_right    = (8.0, 8.0)
        p_bottom_right = (8.0, 2.0)

        m_bottom_left  = (4.0, 4.0)
        m_top_left     = (4.0, 6.0)
        m_top_right    = (6.0, 6.0)
        m_bottom_right = (6.0, 4.0)
        
        # Draw lines from midpoints to propeller centers
        self.draw_line_segment(m_bottom_left[0], m_bottom_left[1], p_bottom_left[0], p_bottom_left[1])
        self.draw_line_segment(m_top_left[0], m_top_left[1], p_top_left[0], p_top_left[1])
        self.draw_line_segment(m_top_right[0], m_top_right[1], p_top_right[0], p_top_right[1])
        self.draw_line_segment(m_bottom_right[0], m_bottom_right[1], p_bottom_right[0], p_bottom_right[1])

        # Return to Center
        self.get_logger().info('Drawing complete. Returning to center.')
        self.call_set_pen_service(0, 0, 0, 2, 1) # Pen off
        self.call_teleport_service(5.0, 5.0, 0.0)

        self.get_logger().info('Drone drawing task finished successfully!')
        # Allow a moment for the final teleport to register before shutdown
        time.sleep(1)
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    drone_controller = DroneController()
    # The node is spun by the drawing logic itself and shutdown at the end
    # No need for an explicit spin call here
    
if __name__ == '__main__':
    main()
