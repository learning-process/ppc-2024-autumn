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
include modules/core/CMakeFiles/core_func_tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include modules/core/CMakeFiles/core_func_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include modules/core/CMakeFiles/core_func_tests.dir/progress.make

# Include the compile flags for this target's objects.
include modules/core/CMakeFiles/core_func_tests.dir/flags.make

modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o: modules/core/CMakeFiles/core_func_tests.dir/flags.make
modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o: modules/core/perf/func_tests/perf_tests.cpp
modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o: modules/core/CMakeFiles/core_func_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/VScode/ppc/ppc-2024-autumn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o"
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o -MF CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o.d -o CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o -c /home/user/VScode/ppc/ppc-2024-autumn/modules/core/perf/func_tests/perf_tests.cpp

modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.i"
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/VScode/ppc/ppc-2024-autumn/modules/core/perf/func_tests/perf_tests.cpp > CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.i

modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.s"
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/VScode/ppc/ppc-2024-autumn/modules/core/perf/func_tests/perf_tests.cpp -o CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.s

modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o: modules/core/CMakeFiles/core_func_tests.dir/flags.make
modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o: modules/core/task/func_tests/task_tests.cpp
modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o: modules/core/CMakeFiles/core_func_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/VScode/ppc/ppc-2024-autumn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o"
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o -MF CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o.d -o CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o -c /home/user/VScode/ppc/ppc-2024-autumn/modules/core/task/func_tests/task_tests.cpp

modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.i"
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/VScode/ppc/ppc-2024-autumn/modules/core/task/func_tests/task_tests.cpp > CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.i

modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.s"
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/VScode/ppc/ppc-2024-autumn/modules/core/task/func_tests/task_tests.cpp -o CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.s

# Object files for target core_func_tests
core_func_tests_OBJECTS = \
"CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o" \
"CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o"

# External object files for target core_func_tests
core_func_tests_EXTERNAL_OBJECTS =

bin/core_func_tests: modules/core/CMakeFiles/core_func_tests.dir/perf/func_tests/perf_tests.cpp.o
bin/core_func_tests: modules/core/CMakeFiles/core_func_tests.dir/task/func_tests/task_tests.cpp.o
bin/core_func_tests: modules/core/CMakeFiles/core_func_tests.dir/build.make
bin/core_func_tests: arch/libcore_module_lib.a
bin/core_func_tests: modules/core/CMakeFiles/core_func_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/user/VScode/ppc/ppc-2024-autumn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../bin/core_func_tests"
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/core_func_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/core/CMakeFiles/core_func_tests.dir/build: bin/core_func_tests
.PHONY : modules/core/CMakeFiles/core_func_tests.dir/build

modules/core/CMakeFiles/core_func_tests.dir/clean:
	cd /home/user/VScode/ppc/ppc-2024-autumn/modules/core && $(CMAKE_COMMAND) -P CMakeFiles/core_func_tests.dir/cmake_clean.cmake
.PHONY : modules/core/CMakeFiles/core_func_tests.dir/clean

modules/core/CMakeFiles/core_func_tests.dir/depend:
	cd /home/user/VScode/ppc/ppc-2024-autumn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/VScode/ppc/ppc-2024-autumn /home/user/VScode/ppc/ppc-2024-autumn/modules/core /home/user/VScode/ppc/ppc-2024-autumn /home/user/VScode/ppc/ppc-2024-autumn/modules/core /home/user/VScode/ppc/ppc-2024-autumn/modules/core/CMakeFiles/core_func_tests.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : modules/core/CMakeFiles/core_func_tests.dir/depend

