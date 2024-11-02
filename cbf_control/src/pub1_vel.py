#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

def pub_vel(pub, dxu):
    # rospy.init_node('pub1_vel', anonymous=True)

    vel = Twist()
    vel.linear.x = dxu[0]*38
    vel.linear.y = 0
    vel.linear.z = 0
    vel.angular.x = 0
    vel.angular.y = 0
    vel.angular.z = dxu[1]
    pub.publish(vel)


