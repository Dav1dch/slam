cmake_minimum_required(VERSION 3.21)

project (flow)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
										"/usr/local/include/eigen3"
	)


add_executable(flow flow.cc)
add_executable(multi_flow multi_flow.cc)

TARGET_LINK_LIBRARIES (flow ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES (multi_flow ${OpenCV_LIBS})
