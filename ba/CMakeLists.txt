cmake_minimum_required(VERSION 3.21)

project (ba)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
										${Sophus_INCLUDE_DIRS}
										"/usr/local/include/eigen3"
	)


add_executable(ba ba.cc)

TARGET_LINK_LIBRARIES (ba ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES (ba ${Sophus_LIBS})
