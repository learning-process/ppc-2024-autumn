# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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
CMAKE_SOURCE_DIR = /home/user/VScode/ppc/ppc-2024-autumn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user/VScode/ppc/ppc-2024-autumn

# Include any dependencies generated for this target.
include tasks/CMakeFiles/stl_module_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tasks/CMakeFiles/stl_module_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include tasks/CMakeFiles/stl_module_lib.dir/progress.make

# Include the compile flags for this target's objects.
include tasks/CMakeFiles/stl_module_lib.dir/flags.make

tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o: tasks/CMakeFiles/stl_module_lib.dir/flags.make
tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o: tasks/stl/example/src/ops_stl.cpp
tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o: tasks/CMakeFiles/stl_module_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/VScode/ppc/ppc-2024-autumn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o -MF CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o.d -o CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o -c /home/user/VScode/ppc/ppc-2024-autumn/tasks/stl/example/src/ops_stl.cpp

tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.i"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/VScode/ppc/ppc-2024-autumn/tasks/stl/example/src/ops_stl.cpp > CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.i

tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.s"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/VScode/ppc/ppc-2024-autumn/tasks/stl/example/src/ops_stl.cpp -o CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.s

# Object files for target stl_module_lib
stl_module_lib_OBJECTS = \
"CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o"

# External object files for target stl_module_lib
stl_module_lib_EXTERNAL_OBJECTS =

arch/libstl_module_lib.a: tasks/CMakeFiles/stl_module_lib.dir/stl/example/src/ops_stl.cpp.o
arch/libstl_module_lib.a: tasks/CMakeFiles/stl_module_lib.dir/build.make
arch/libstl_module_lib.a: tasks/CMakeFiles/stl_module_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/user/VScode/ppc/ppc-2024-autumn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../arch/libstl_module_lib.a"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && $(CMAKE_COMMAND) -P CMakeFiles/stl_module_lib.dir/cmake_clean_target.cmake
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stl_module_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tasks/CMakeFiles/stl_module_lib.dir/build: arch/libstl_module_lib.a
.PHONY : tasks/CMakeFiles/stl_module_lib.dir/build

tasks/CMakeFiles/stl_module_lib.dir/clean:
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && $(CMAKE_COMMAND) -P CMakeFiles/stl_module_lib.dir/cmake_clean.cmake
.PHONY : tasks/CMakeFiles/stl_module_lib.dir/clean

tasks/CMakeFiles/stl_module_lib.dir/depend:
	cd /home/user/VScode/ppc/ppc-2024-autumn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/VScode/ppc/ppc-2024-autumn /home/user/VScode/ppc/ppc-2024-autumn/tasks /home/user/VScode/ppc/ppc-2024-autumn /home/user/VScode/ppc/ppc-2024-autumn/tasks /home/user/VScode/ppc/ppc-2024-autumn/tasks/CMakeFiles/stl_module_lib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tasks/CMakeFiles/stl_module_lib.dir/depend

