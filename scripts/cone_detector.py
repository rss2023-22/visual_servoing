#!/usr/bin/env python

import numpy as np
import rospy

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from visual_servoing.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


class ConeDetector():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    LINE_FOLLOWING = 1.0 #rospy.get_param("visual_servoing/line_following")
    TESTING = False # to pass to color_segmentation for minimal latency and no visualization
    LOW_BOUND = 225
    HIGH_BOUND = 275

    def __init__(self):
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.cone_pub = rospy.Publisher("/relative_cone_px", ConeLocationPixel, queue_size=10)
        self.debug_pub = rospy.Publisher("/cone_debug_img", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8") # is this right? convert msg to cv2 format. 
        bounding_box = cd_color_segmentation(img,None,self.LINE_FOLLOWING,self.TESTING)  # (x,y),(x+w,y+h)
        noCone = ((0,0),(0,0))
             

        # create ConeLocationPixel object and publish
        coneLoc = ConeLocationPixel()
        if noCone == bounding_box:
            coneLoc.u = -1000.0
            coneLoc.v = -1000.0
        else:
            coneLoc.u = float(int((bounding_box[0][0]+bounding_box[1][0])/2)) # I assume u=x?
            coneLoc.v = bounding_box[1][1] # I assume v=y?
        self.cone_pub.publish(coneLoc)

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)


if __name__ == '__main__':
    try:
        rospy.init_node('ConeDetector', anonymous=True)
        ConeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
