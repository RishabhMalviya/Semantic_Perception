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
include CMakeFiles/kNN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kNN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kNN.dir/flags.make

CMakeFiles/kNN.dir/src/kNN.cpp.o: CMakeFiles/kNN.dir/flags.make
CMakeFiles/kNN.dir/src/kNN.cpp.o: ../src/kNN.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/kNN.dir/src/kNN.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kNN.dir/src/kNN.cpp.o -c /home/revinci/code/workspaces/qt_ws/Semantic_Perception/src/kNN.cpp

CMakeFiles/kNN.dir/src/kNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kNN.dir/src/kNN.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/revinci/code/workspaces/qt_ws/Semantic_Perception/src/kNN.cpp > CMakeFiles/kNN.dir/src/kNN.cpp.i

CMakeFiles/kNN.dir/src/kNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kNN.dir/src/kNN.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/revinci/code/workspaces/qt_ws/Semantic_Perception/src/kNN.cpp -o CMakeFiles/kNN.dir/src/kNN.cpp.s

CMakeFiles/kNN.dir/src/kNN.cpp.o.requires:
.PHONY : CMakeFiles/kNN.dir/src/kNN.cpp.o.requires

CMakeFiles/kNN.dir/src/kNN.cpp.o.provides: CMakeFiles/kNN.dir/src/kNN.cpp.o.requires
	$(MAKE) -f CMakeFiles/kNN.dir/build.make CMakeFiles/kNN.dir/src/kNN.cpp.o.provides.build
.PHONY : CMakeFiles/kNN.dir/src/kNN.cpp.o.provides

CMakeFiles/kNN.dir/src/kNN.cpp.o.provides.build: CMakeFiles/kNN.dir/src/kNN.cpp.o

# Object files for target kNN
kNN_OBJECTS = \
"CMakeFiles/kNN.dir/src/kNN.cpp.o"

# External object files for target kNN
kNN_EXTERNAL_OBJECTS =

kNN: CMakeFiles/kNN.dir/src/kNN.cpp.o
kNN: CMakeFiles/kNN.dir/build.make
kNN: libVigraImpex.a
kNN: /usr/local/lib/libmlpack.so
kNN: /usr/lib/liblapack.so
kNN: /usr/lib/libf77blas.so
kNN: /usr/lib/libatlas.so
kNN: /usr/lib/libf77blas.so
kNN: /usr/lib/libatlas.so
kNN: CMakeFiles/kNN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable kNN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kNN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kNN.dir/build: kNN
.PHONY : CMakeFiles/kNN.dir/build

CMakeFiles/kNN.dir/requires: CMakeFiles/kNN.dir/src/kNN.cpp.o.requires
.PHONY : CMakeFiles/kNN.dir/requires

CMakeFiles/kNN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kNN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kNN.dir/clean

CMakeFiles/kNN.dir/depend:
	cd /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/revinci/code/workspaces/qt_ws/Semantic_Perception /home/revinci/code/workspaces/qt_ws/Semantic_Perception /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build /home/revinci/code/workspaces/qt_ws/Semantic_Perception/build/CMakeFiles/kNN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kNN.dir/depend
