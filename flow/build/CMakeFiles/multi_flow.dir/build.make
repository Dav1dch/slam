# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/latte/Codes/slam/flow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/latte/Codes/slam/flow/build

# Include any dependencies generated for this target.
include CMakeFiles/multi_flow.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/multi_flow.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/multi_flow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/multi_flow.dir/flags.make

CMakeFiles/multi_flow.dir/multi_flow.cc.o: CMakeFiles/multi_flow.dir/flags.make
CMakeFiles/multi_flow.dir/multi_flow.cc.o: ../multi_flow.cc
CMakeFiles/multi_flow.dir/multi_flow.cc.o: CMakeFiles/multi_flow.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/latte/Codes/slam/flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/multi_flow.dir/multi_flow.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/multi_flow.dir/multi_flow.cc.o -MF CMakeFiles/multi_flow.dir/multi_flow.cc.o.d -o CMakeFiles/multi_flow.dir/multi_flow.cc.o -c /home/latte/Codes/slam/flow/multi_flow.cc

CMakeFiles/multi_flow.dir/multi_flow.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multi_flow.dir/multi_flow.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/latte/Codes/slam/flow/multi_flow.cc > CMakeFiles/multi_flow.dir/multi_flow.cc.i

CMakeFiles/multi_flow.dir/multi_flow.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multi_flow.dir/multi_flow.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/latte/Codes/slam/flow/multi_flow.cc -o CMakeFiles/multi_flow.dir/multi_flow.cc.s

# Object files for target multi_flow
multi_flow_OBJECTS = \
"CMakeFiles/multi_flow.dir/multi_flow.cc.o"

# External object files for target multi_flow
multi_flow_EXTERNAL_OBJECTS =

multi_flow: CMakeFiles/multi_flow.dir/multi_flow.cc.o
multi_flow: CMakeFiles/multi_flow.dir/build.make
multi_flow: /usr/local/lib/libopencv_dnn.so.3.4.16
multi_flow: /usr/local/lib/libopencv_highgui.so.3.4.16
multi_flow: /usr/local/lib/libopencv_ml.so.3.4.16
multi_flow: /usr/local/lib/libopencv_objdetect.so.3.4.16
multi_flow: /usr/local/lib/libopencv_shape.so.3.4.16
multi_flow: /usr/local/lib/libopencv_stitching.so.3.4.16
multi_flow: /usr/local/lib/libopencv_superres.so.3.4.16
multi_flow: /usr/local/lib/libopencv_videostab.so.3.4.16
multi_flow: /usr/local/lib/libopencv_calib3d.so.3.4.16
multi_flow: /usr/local/lib/libopencv_features2d.so.3.4.16
multi_flow: /usr/local/lib/libopencv_flann.so.3.4.16
multi_flow: /usr/local/lib/libopencv_photo.so.3.4.16
multi_flow: /usr/local/lib/libopencv_video.so.3.4.16
multi_flow: /usr/local/lib/libopencv_videoio.so.3.4.16
multi_flow: /usr/local/lib/libopencv_imgcodecs.so.3.4.16
multi_flow: /usr/local/lib/libopencv_imgproc.so.3.4.16
multi_flow: /usr/local/lib/libopencv_core.so.3.4.16
multi_flow: CMakeFiles/multi_flow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/latte/Codes/slam/flow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable multi_flow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multi_flow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/multi_flow.dir/build: multi_flow
.PHONY : CMakeFiles/multi_flow.dir/build

CMakeFiles/multi_flow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/multi_flow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/multi_flow.dir/clean

CMakeFiles/multi_flow.dir/depend:
	cd /home/latte/Codes/slam/flow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/latte/Codes/slam/flow /home/latte/Codes/slam/flow /home/latte/Codes/slam/flow/build /home/latte/Codes/slam/flow/build /home/latte/Codes/slam/flow/build/CMakeFiles/multi_flow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/multi_flow.dir/depend

