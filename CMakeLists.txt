cmake_minimum_required(VERSION 3.10)
project(Perception VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV_DIR for Homebrew installation
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.11.0_1/lib/cmake/opencv4")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(
    Perception
    feature_detector.cpp 
    main.cpp
)

# Link OpenCV libraries
target_link_libraries(Perception ${OpenCV_LIBS})

