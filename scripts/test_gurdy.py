#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
import time

def test_controllers():
    rospy.init_node('test_controllers', anonymous=True)
    
    # Create publishers for all six upper leg joints
    publishers = []
    for i in range(1, 7):
        pub = rospy.Publisher(f'/gurdy/head_upperlegM{i}_joint_position_controller/command', 
                             Float64, queue_size=1)
        publishers.append(pub)
    
    # Wait for connections
    rospy.loginfo("Waiting for connections...")
    time.sleep(2)
    
    # Simple movement test pattern
    rospy.loginfo("Starting movement test")
    rate = rospy.Rate(1)  # 1 Hz
    
    for _ in range(5):  # Run for 5 cycles
        # Move all legs up
        for pub in publishers:
            pub.publish(Float64(-0.5))
        rospy.loginfo("Legs up")
        rate.sleep()
        
        # Move all legs down
        for pub in publishers:
            pub.publish(Float64(0.0))
        rospy.loginfo("Legs down")
        rate.sleep()
    
    rospy.loginfo("Test complete")

if __name__ == '__main__':
    try:
        test_controllers()
    except rospy.ROSInterruptException:
        pass