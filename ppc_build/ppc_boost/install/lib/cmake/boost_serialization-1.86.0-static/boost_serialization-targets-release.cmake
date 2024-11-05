#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Boost::serialization" for configuration "Release"
set_property(TARGET Boost::serialization APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Boost::serialization PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libboost_serialization-vc143-mt-x64-1_86.lib"
  )

list(APPEND _cmake_import_check_targets Boost::serialization )
list(APPEND _cmake_import_check_files_for_Boost::serialization "${_IMPORT_PREFIX}/lib/libboost_serialization-vc143-mt-x64-1_86.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
