#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Boost::random" for configuration "Release"
set_property(TARGET Boost::random APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Boost::random PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libboost_random-vc143-mt-x64-1_86.lib"
  )

list(APPEND _cmake_import_check_targets Boost::random )
list(APPEND _cmake_import_check_files_for_Boost::random "${_IMPORT_PREFIX}/lib/libboost_random-vc143-mt-x64-1_86.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
