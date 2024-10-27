message(STATUS "Modules")

SUBDIRLIST(subdirs ${CMAKE_CURRENT_SOURCE_DIR})

foreach(subd ${subdirs})
  if (NOT (${subd} STREQUAL "CMakeFiles"))
    add_subdirectory(${subd})
  endif ()
endforeach()
