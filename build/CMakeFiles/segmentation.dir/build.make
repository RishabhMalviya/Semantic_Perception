# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/revinci/code/workspaces/qt_ws/Semantic_Perception

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build

# Include any dependencies generated for this target.
include CMakeFiles/segmentation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/segmentation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/segmentation.dir/flags.make

CMakeFiles/segmentation.dir/src/segmentation.cpp.o: CMakeFiles/segmentation.dir/flags.make
CMakeFiles/segmentation.dir/src/segmentation.cpp.o: ../src/segmentation.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/segmentation.dir/src/segmentation.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/segmentation.dir/src/segmentation.cpp.o -c /home/revinci/code/workspaces/qt_ws/Semantic_Perception/src/segmentation.cpp

CMakeFiles/segmentation.dir/src/segmentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segmentation.dir/src/segmentation.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/revinci/code/workspaces/qt_ws/Semantic_Perception/src/segmentation.cpp > CMakeFiles/segmentation.dir/src/segmentation.cpp.i

CMakeFiles/segmentation.dir/src/segmentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segmentation.dir/src/segmentation.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/revinci/code/workspaces/qt_ws/Semantic_Perception/src/segmentation.cpp -o CMakeFiles/segmentation.dir/src/segmentation.cpp.s

CMakeFiles/segmentation.dir/src/segmentation.cpp.o.requires:
.PHONY : CMakeFiles/segmentation.dir/src/segmentation.cpp.o.requires

CMakeFiles/segmentation.dir/src/segmentation.cpp.o.provides: CMakeFiles/segmentation.dir/src/segmentation.cpp.o.requires
	$(MAKE) -f CMakeFiles/segmentation.dir/build.make CMakeFiles/segmentation.dir/src/segmentation.cpp.o.provides.build
.PHONY : CMakeFiles/segmentation.dir/src/segmentation.cpp.o.provides

CMakeFiles/segmentation.dir/src/segmentation.cpp.o.provides.build: CMakeFiles/segmentation.dir/src/segmentation.cpp.o

# Object files for target segmentation
segmentation_OBJECTS = \
"CMakeFiles/segmentation.dir/src/segmentation.cpp.o"

# External object files for target segmentation
segmentation_EXTERNAL_OBJECTS =

segmentation: CMakeFiles/segmentation.dir/src/segmentation.cpp.o
segmentation: CMakeFiles/segmentation.dir/build.make
segmentation: libVigraImpex.a
segmentation: CMakeFiles/segmentation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable segmentation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/segmentation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/segmentation.dir/build: segmentation
.PHONY : CMakeFiles/segmentation.dir/build

CMakeFiles/segmentation.dir/requires: CMakeFiles/segmentation.dir/src/segmentation.cpp.o.requires
.PHONY : CMakeFiles/segmentation.dir/requires

CMakeFiles/segmentation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/segmentation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/segmentation.dir/clean

CMakeFiles/segmentation.dir/depend:
	cd /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/revinci/code/workspaces/qt_ws/Semantic_Perception /home/revinci/code/workspaces/qt_ws/Semantic_Perception /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build/CMakeFiles/segmentation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/segmentation.dir/depend
