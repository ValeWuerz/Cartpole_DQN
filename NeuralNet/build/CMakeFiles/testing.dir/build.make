# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/valentin/Master_Thesis/Coding/NeuralNet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/valentin/Master_Thesis/Coding/NeuralNet/build

# Include any dependencies generated for this target.
include CMakeFiles/testing.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testing.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testing.dir/flags.make

CMakeFiles/testing.dir/testing.cpp.o: CMakeFiles/testing.dir/flags.make
CMakeFiles/testing.dir/testing.cpp.o: ../testing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/valentin/Master_Thesis/Coding/NeuralNet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testing.dir/testing.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testing.dir/testing.cpp.o -c /home/valentin/Master_Thesis/Coding/NeuralNet/testing.cpp

CMakeFiles/testing.dir/testing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testing.dir/testing.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/valentin/Master_Thesis/Coding/NeuralNet/testing.cpp > CMakeFiles/testing.dir/testing.cpp.i

CMakeFiles/testing.dir/testing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testing.dir/testing.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/valentin/Master_Thesis/Coding/NeuralNet/testing.cpp -o CMakeFiles/testing.dir/testing.cpp.s

# Object files for target testing
testing_OBJECTS = \
"CMakeFiles/testing.dir/testing.cpp.o"

# External object files for target testing
testing_EXTERNAL_OBJECTS =

testing: CMakeFiles/testing.dir/testing.cpp.o
testing: CMakeFiles/testing.dir/build.make
testing: /home/valentin/Cpp_pack/libtorch/lib/libtorch.so
testing: /home/valentin/Cpp_pack/libtorch/lib/libc10.so
testing: /home/valentin/Cpp_pack/libtorch/lib/libkineto.a
testing: /home/valentin/Cpp_pack/libtorch/lib/libc10.so
testing: CMakeFiles/testing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/valentin/Master_Thesis/Coding/NeuralNet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testing"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testing.dir/build: testing

.PHONY : CMakeFiles/testing.dir/build

CMakeFiles/testing.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testing.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testing.dir/clean

CMakeFiles/testing.dir/depend:
	cd /home/valentin/Master_Thesis/Coding/NeuralNet/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Master_Thesis/Coding/NeuralNet /home/valentin/Master_Thesis/Coding/NeuralNet /home/valentin/Master_Thesis/Coding/NeuralNet/build /home/valentin/Master_Thesis/Coding/NeuralNet/build /home/valentin/Master_Thesis/Coding/NeuralNet/build/CMakeFiles/testing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testing.dir/depend

