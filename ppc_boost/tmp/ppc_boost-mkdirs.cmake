# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/user/VScode/ppc/ppc-2024-autumn/3rdparty/boost")
  file(MAKE_DIRECTORY "/home/user/VScode/ppc/ppc-2024-autumn/3rdparty/boost")
endif()
file(MAKE_DIRECTORY
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/build"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/install"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/tmp"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/src"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/user/VScode/ppc/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp${cfgdir}") # cfgdir has leading slash
endif()
