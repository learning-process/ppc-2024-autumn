# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/user/VScode/ppc/ppc-2024-autumn/3rdparty/onetbb")
  file(MAKE_DIRECTORY "/home/user/VScode/ppc/ppc-2024-autumn/3rdparty/onetbb")
endif()
file(MAKE_DIRECTORY
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/build"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/install"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/tmp"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/src/ppc_onetbb-stamp"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/src"
  "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/src/ppc_onetbb-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/src/ppc_onetbb-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/user/VScode/ppc/ppc-2024-autumn/ppc_onetbb/src/ppc_onetbb-stamp${cfgdir}") # cfgdir has leading slash
endif()
