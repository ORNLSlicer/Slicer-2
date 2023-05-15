#add_definitions(-W4)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "")
set(ASSIMP_BUILD_ASSIMP_VIEW OFF CACHE BOOL "")
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "")

add_subdirectory(clipper)
add_subdirectory(zip)
add_subdirectory(psimpl)
add_subdirectory(qtxlsxwriter)

# Set Parent's includes
list(APPEND libs_include "contrib/json-fifo_map" ${Boost_INCLUDE_DIRS} "contrib/qtxlsxwriter/src/xlsx")
set(libs_include ${libs_include} PARENT_SCOPE)

# Set Parent's libs
list(APPEND libs polyclipping psimpl zip ${Boost_LIBRARIES} QtXlsxWriter)
set(libs ${libs} PARENT_SCOPE)

# Single Path Library
find_library(SINGLE_PATH_LIBRARY single_path "single_path")
if (SINGLE_PATH_LIBRARY)
    set(SINGLE_PATH_FOUND true PARENT_SCOPE)

    # Set Parent's Include
    list(APPEND libs_include "contrib/single_path/include")
    set(libs_include ${libs_include} PARENT_SCOPE)

    # Set Parent's Libs
    list(APPEND libs ${SINGLE_PATH_LIBRARY})
    set(libs ${libs} PARENT_SCOPE)
endif ()

# Wire Feed Library
find_library(WIRE_FEED_LIBRARY wire_feed "wire_feed")
if (WIRE_FEED_LIBRARY)
    set(WIRE_FEED_FOUND true PARENT_SCOPE)

    # Set Parent's Include
    list(APPEND libs_include "contrib/wire_feed/include")
    set(libs_include ${libs_include} PARENT_SCOPE)

    # Set Parent's Libs
    list(APPEND libs ${WIRE_FEED_LIBRARY})
    set(libs ${libs} PARENT_SCOPE)
endif ()
