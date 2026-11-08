cmake_minimum_required(VERSION 3.0.2)
project(my_gurdy_description)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

## Declare custom messages
add_message_files(
  FILES
  a.msg
)

## Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare the catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

