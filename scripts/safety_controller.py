#!/usr/bin/env python2

import numpy as np

import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
#from visualization_msgs.msg import Marker
#from visualization_tools import *
import tf2_ros

class SafetyController:
    
    '''
    SafetyController. Manges priorities and stops car if collision imminent. 
    Params:
        - User Params:
            - pause_time: float, amount of time to pause before moving.
                - if inf: never moves till node restarted
                - if 0: moves after obstacle removed
                - if x>0: moves at x seconds after obstacle removed
        - Other vars
            - min_dist: active update for minimum distance
            - paused: tracks if driving stopped
            - start_time: time of last time obstacle sensed
    '''

    # USER INPUTS:
    pause_time = 3.0 # seconds

    # ACTIVE STUFF
    min_dist = None 
    paused = False # tracks if driving commands are currently paused
    start_time = None # tracks start_time (time when pause command active)

    def __init__(self):
        
        '''
        Initialize publishers ans subscribers
        '''
        
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/safety',AckermannDriveStamped,queue_size=10)
        rospy.Subscriber('/scan',LaserScan,self.read_laser_scan)
        rospy.Subscriber('/vesc/high_level/ackermann_cmd_mux/output',AckermannDriveStamped,self.check_drive_command)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def read_laser_scan(self,data):

        '''
        Processes laserscan, extracts forwards data
            - Input: data, laserscan object
            - Output: None, updates self.min_dist 
        '''

        arrVals = np.array(data.ranges)
        portion = len(arrVals)/10.0
        forwards = arrVals[int(4*portion):int(5*portion)]
        self.min_dist = min(forwards)
    
    def check_drive_command(self,data):

        '''
        Callback function. Publishes drive command, modifies drive command so
        the car stops if a collission is imminent. 
            - Input: data, callback data
            - Output: none, publishes drive commnd
        '''

        if self.start_time != None:
            dt = (self.start_time - rospy.Time.now())
            dt = dt.to_sec()
        if self.paused and abs(dt) > self.pause_time:
            self.paused = False 

        if self.min_dist == None or self.min_dist < 0.15*data.drive.speed+0.30 or self.paused:
            drive_command = AckermannDriveStamped()
            drive_command.header.frame_id = 'base_link'
            drive_command.header.stamp = rospy.Time()
            drive_command.drive.speed = 0
            self.drive_pub.publish(drive_command)
            if not self.paused:
                self.paused = True
                self.start_time = rospy.Time.now()
            if self.min_dist == None or self.min_dist < 0.15*data.drive.speed+0.30:
                self.start_time = rospy.Time.now()
        else:
            self.drive_pub.publish(data)


if __name__ == "__main__":
    rospy.init_node('safety_controller')
    safety_controller = SafetyController()
    rospy.spin()
