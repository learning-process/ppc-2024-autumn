# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/3rdparty/boost")
  file(MAKE_DIRECTORY "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/3rdparty/boost")
endif()
file(MAKE_DIRECTORY
  "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/build"
  "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/install"
  "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/tmp"
  "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp"
  "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/src"
  "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/Elena/Documents/GitHub/ppc-2024-autumn/ppc_boost/src/ppc_boost-stamp${cfgdir}") # cfgdir has leading slash
endif()
