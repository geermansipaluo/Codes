import serial
# This script open a connection with 2 robots and exchanged data with them using the "advanced sercom protocol".
# Tested with Python 3.4
# Requirements: PySerial module.

import time
import struct

robot1 = serial.Serial('COM132', 115200, timeout=0)  # Specify the robot communication port (after pairing).
robot2 = serial.Serial('COM128', 115200, timeout=0)  # Specify the robot communication port (after pairing).

# Robot 1 actuators state.
rob1_leds = bytearray([0, 0, 0, 0, 0, 0, 0, 0])
rob1_left_speed = 0
rob1_right_speed = 0

# Robot 2 actuators state.
rob2_leds = bytearray([1, 1, 1, 1, 1, 1, 1, 1])
rob2_left_speed = 0
rob2_right_speed = 0

while (1):
    # Prepare the message to send to robot1.
    left = struct.unpack('bb', struct.pack('h', rob1_left_speed))
    right = struct.unpack('bb', struct.pack('h', rob1_right_speed))
    message = struct.pack("<bbbbbbbbbbbbb", - ord('a'), - ord('N'), - ord('g'), - ord('u'), - ord('L'), 0, rob1_leds[0],
                          - ord('D'), left[0], left[1], right[0], right[1], 0)
    robot1.write(message)
    # Read the expected data from robot1.
    reply = robot1.read()
    while len(reply) < 34:
        reply += robot1.read()
    # Extract the sensors data.
    rob1_acc = struct.unpack('@hhh', reply[0:6])
    rob1_prox = struct.unpack('@HHHHHHHH', reply[6:22])
    rob1_gyro = struct.unpack('@hhh', reply[22:28])
    rob1_micro = struct.unpack('@HHH', reply[28:34])
    print("\r\nrob1")
    print("acc0: " + str(rob1_acc[0]))
    print("prox0: " + str(rob1_prox[0]))
    print("gyro0: " + str(rob1_gyro[0]))
    print("micro0: " + str(rob1_micro[0]))

    # Prepare the message to send to robot1.
    left = struct.unpack('bb', struct.pack('h', rob2_left_speed))
    right = struct.unpack('bb', struct.pack('h', rob2_right_speed))
    message = struct.pack("<bbbbbbbbbbbbb", - ord('a'), - ord('N'), - ord('g'), - ord('u'), - ord('L'), 0, rob2_leds[0],
                          - ord('D'), left[0], left[1], right[0], right[1], 0)
    robot2.write(message)
    # Read the expected data from robot1.
    reply = robot2.read()
    while len(reply) < 34:
        reply += robot2.read()
    # Extract the sensors data.
    rob2_acc = struct.unpack('@hhh', reply[0:6])
    rob2_prox = struct.unpack('@HHHHHHHH', reply[6:22])
    rob2_gyro = struct.unpack('@hhh', reply[22:28])
    rob2_micro = struct.unpack('@HHH', reply[28:34])
    print("\r\nrob2")
    print("acc0: " + str(rob2_acc[0]))
    print("prox0: " + str(rob2_prox[0]))
    print("gyro0: " + str(rob2_gyro[0]))
    print("micro0: " + str(rob2_micro[0]))

    # Your behavior...
    if rob2_prox[0] > 2000:
        rob1_leds = bytearray([1, 1, 1, 1, 1, 1, 1, 1])
        rob2_leds = bytearray([0, 0, 0, 0, 0, 0, 0, 0])
    else:
        rob1_leds = bytearray([0, 0, 0, 0, 0, 0, 0, 0])
        rob2_leds = bytearray([1, 1, 1, 1, 1, 1, 1, 1])

robot1.close()
robot2.close()

