#!/usr/bin/env python

import rospy
import numpy as np

from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        LINE_FOLLOWING = 1 #rospy.get_param("visual_servoing/line_following")
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            ParkingError, queue_size=10)

        if LINE_FOLLOWING == 1:
            self.parking_distance = .3 # meters; try playing with this number!
            self.turning_radius = 0.4 #turning radius of the car
        else:
            self.parking_distance = .75 #rospy.get_param("visual_servoing/parking_distance")
            self.turning_radius = 0.9
        
        self.relative_x = 0
        self.relative_y = 0
        self.angle_tolerance = 0.1
        self.distance_tolerance = 0.07
        
        self.drive_speed = 1
        self.max_steering_angle = 0.34
        self.forward = None

    def relative_cone_callback(self, msg):
        '''
        Callback when a new cone position is received.
        
        msg has two float32 (x_pos, y_pos); x+ is forward, y+ is to the left
        When x_pos and y_pos are both exactly zero, the cone is assumed to not be in frame
        In this case, the car will do nothing
        '''
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        
        drive_cmd = AckermannDriveStamped()

        #################################
        
        relative_angle = np.arctan2(self.relative_y,self.relative_x)
        relative_distance = (self.relative_x**2+self.relative_y**2)**0.5
        
        drive_cmd.header.frame_id = 'base_link'
        drive_cmd.header.stamp = rospy.Time()
        
        adj_drive_speed = min(1,max(1.2*relative_distance/self.parking_distance-1,0.2))*self.drive_speed
        
        if relative_distance == 0: #no cone found
            drive_cmd.drive.speed = 0
        
        elif abs(relative_angle) < self.angle_tolerance: #car is approximately aligned
            if abs(relative_distance-self.parking_distance) < self.distance_tolerance:
                drive_cmd.drive.speed = 0 #car is parked within tolerance
            else: #car needs to drive forward or backward
                sign = 1.0 if relative_distance>self.parking_distance else -1.0
                drive_cmd.drive.speed = sign*adj_drive_speed
                drive_cmd.drive.steering_angle = relative_angle
                
        else:
            if self.forward == None: self.forward = relative_distance>self.parking_distance
            
            if (self.forward == True and relative_distance < 0.8*self.parking_distance) or \
               (self.forward == False and relative_distance > 1.2*self.parking_distance):
                self.forward = not self.forward
            
            if self.forward == False or 2*self.turning_radius*np.sin(relative_angle) > relative_distance:
                sign = -1 #too close to go forward, must reverse first
            else: sign = 1

            drive_cmd.drive.speed = sign*adj_drive_speed
            drive_cmd.drive.steering_angle = sign*np.sign(relative_angle)*min(abs(relative_angle),self.max_steering_angle)

        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = (self.relative_x**2+self.relative_y**2)**0.5

        #################################
        
        self.error_pub.publish(error_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('ParkingController', anonymous=True)
        ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
