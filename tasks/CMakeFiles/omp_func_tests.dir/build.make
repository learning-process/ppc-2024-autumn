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
include tasks/CMakeFiles/omp_func_tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tasks/CMakeFiles/omp_func_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include tasks/CMakeFiles/omp_func_tests.dir/progress.make

# Include the compile flags for this target's objects.
include tasks/CMakeFiles/omp_func_tests.dir/flags.make

tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o: tasks/CMakeFiles/omp_func_tests.dir/flags.make
tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o: tasks/omp/example/func_tests/main.cpp
tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o: tasks/CMakeFiles/omp_func_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/VScode/ppc/ppc-2024-autumn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o -MF CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o.d -o CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o -c /home/user/VScode/ppc/ppc-2024-autumn/tasks/omp/example/func_tests/main.cpp

tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.i"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/VScode/ppc/ppc-2024-autumn/tasks/omp/example/func_tests/main.cpp > CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.i

tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.s"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/VScode/ppc/ppc-2024-autumn/tasks/omp/example/func_tests/main.cpp -o CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.s

# Object files for target omp_func_tests
omp_func_tests_OBJECTS = \
"CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o"

# External object files for target omp_func_tests
omp_func_tests_EXTERNAL_OBJECTS =

bin/omp_func_tests: tasks/CMakeFiles/omp_func_tests.dir/omp/example/func_tests/main.cpp.o
bin/omp_func_tests: tasks/CMakeFiles/omp_func_tests.dir/build.make
bin/omp_func_tests: arch/libcore_module_lib.a
bin/omp_func_tests: arch/libomp_module_lib.a
bin/omp_func_tests: tasks/CMakeFiles/omp_func_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/user/VScode/ppc/ppc-2024-autumn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/omp_func_tests"
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/omp_func_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tasks/CMakeFiles/omp_func_tests.dir/build: bin/omp_func_tests
.PHONY : tasks/CMakeFiles/omp_func_tests.dir/build

tasks/CMakeFiles/omp_func_tests.dir/clean:
	cd /home/user/VScode/ppc/ppc-2024-autumn/tasks && $(CMAKE_COMMAND) -P CMakeFiles/omp_func_tests.dir/cmake_clean.cmake
.PHONY : tasks/CMakeFiles/omp_func_tests.dir/clean

tasks/CMakeFiles/omp_func_tests.dir/depend:
	cd /home/user/VScode/ppc/ppc-2024-autumn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/VScode/ppc/ppc-2024-autumn /home/user/VScode/ppc/ppc-2024-autumn/tasks /home/user/VScode/ppc/ppc-2024-autumn /home/user/VScode/ppc/ppc-2024-autumn/tasks /home/user/VScode/ppc/ppc-2024-autumn/tasks/CMakeFiles/omp_func_tests.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tasks/CMakeFiles/omp_func_tests.dir/depend

