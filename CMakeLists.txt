cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(2D_feature_tracking)

# Apply -Wall to all targets in this directory
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Use the manually built OpenCV from /tmp/opencv
set(OpenCV_DIR /tmp/opencv/build)
set(CMAKE_PREFIX_PATH /tmp/opencv/build ${CMAKE_PREFIX_PATH})

find_package(OpenCV 4.2 QUIET PATHS /tmp/opencv/build NO_DEFAULT_PATH)

if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV from /tmp not found, using system OpenCV")
    find_package(OpenCV 4.1 REQUIRED)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/tmp/opencv_contrib/modules/xfeatures2d/include)
include_directories(/tmp/opencv/include)

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(/tmp/opencv/build/lib)
add_definitions(${OpenCV_DEFINITIONS})

# Main executable
add_executable(2D_feature_tracking src/matching2D.cpp src/main.cpp)

# Require C++17 scoped to this target (replaces the old global add_definitions)
target_compile_features(2D_feature_tracking PRIVATE cxx_std_17)

# Link to xfeatures2d if available
find_library(XFEATURES2D_LIB opencv_xfeatures2d PATHS /tmp/opencv/build/lib NO_DEFAULT_PATH)
if(XFEATURES2D_LIB)
    message(STATUS "Found xfeatures2d library: ${XFEATURES2D_LIB}")
    target_link_libraries (2D_feature_tracking ${OpenCV_LIBRARIES} ${XFEATURES2D_LIB})
else()
    message(STATUS "xfeatures2d library not found, linking without it")
    target_link_libraries (2D_feature_tracking ${OpenCV_LIBRARIES})
endif()
