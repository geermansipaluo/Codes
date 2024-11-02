#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

def pub_vel(pub, dxu):

    vel = Twist()
    vel.linear.x = dxu[0]*35
    vel.linear.y = 0
    vel.linear.z = 0
    vel.angular.x = 0
    vel.angular.y = 0
    vel.angular.z = dxu[1]*0.7
    pub.publish(vel)


