# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/Laba_ppc/ppc-2024-autumn/3rdparty/googletest")
  file(MAKE_DIRECTORY "D:/Laba_ppc/ppc-2024-autumn/3rdparty/googletest")
endif()
file(MAKE_DIRECTORY
  "D:/Laba_ppc/ppc_build/ppc_googletest/build"
  "D:/Laba_ppc/ppc_build/ppc_googletest/install"
  "D:/Laba_ppc/ppc_build/ppc_googletest/tmp"
  "D:/Laba_ppc/ppc_build/ppc_googletest/src/ppc_googletest-stamp"
  "D:/Laba_ppc/ppc_build/ppc_googletest/src"
  "D:/Laba_ppc/ppc_build/ppc_googletest/src/ppc_googletest-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/Laba_ppc/ppc_build/ppc_googletest/src/ppc_googletest-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/Laba_ppc/ppc_build/ppc_googletest/src/ppc_googletest-stamp${cfgdir}") # cfgdir has leading slash
endif()
