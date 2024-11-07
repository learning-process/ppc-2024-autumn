# CMake generated Testfile for 
# Source directory: D:/Laba_ppc/ppc-2024-autumn/modules/ref
# Build directory: D:/Laba_ppc/ppc_build/modules/ref
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[ref_func_tests]=] "D:/Laba_ppc/ppc_build/bin/ref_func_tests.exe")
  set_tests_properties([=[ref_func_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;44;add_test;D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[ref_func_tests]=] "D:/Laba_ppc/ppc_build/bin/ref_func_tests.exe")
  set_tests_properties([=[ref_func_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;44;add_test;D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[ref_func_tests]=] "D:/Laba_ppc/ppc_build/bin/MinSizeRel/ref_func_tests.exe")
  set_tests_properties([=[ref_func_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;44;add_test;D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[ref_func_tests]=] "D:/Laba_ppc/ppc_build/bin/RelWithDebInfo/ref_func_tests.exe")
  set_tests_properties([=[ref_func_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;44;add_test;D:/Laba_ppc/ppc-2024-autumn/modules/ref/CMakeLists.txt;0;")
else()
  add_test([=[ref_func_tests]=] NOT_AVAILABLE)
endif()
