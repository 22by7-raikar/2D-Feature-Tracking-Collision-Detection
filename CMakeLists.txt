cmake_minimum_required(VERSION 3.10)
project(2D_feature_tracking CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs features2d flann
                     OPTIONAL_COMPONENTS xfeatures2d)

add_executable(2D_feature_tracking src/main.cpp src/features.cpp src/ttc.cpp)
target_compile_features(2D_feature_tracking PRIVATE cxx_std_17)
target_compile_options(2D_feature_tracking PRIVATE -Wall -Wextra)
target_include_directories(2D_feature_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(2D_feature_tracking PRIVATE ${OpenCV_LIBS})

# Default data/output location is the project root, so the binary can be run from anywhere.
target_compile_definitions(2D_feature_tracking PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
    HAS_XFEATURES2D=$<BOOL:${OpenCV_xfeatures2d_FOUND}>)

if(OpenCV_xfeatures2d_FOUND)
    message(STATUS "xfeatures2d found: BRIEF and FREAK descriptors enabled")
else()
    message(STATUS "xfeatures2d not found: BRIEF and FREAK descriptors disabled")
endif()
