cmake_minimum_required(VERSION 3.1)

project(PRO)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(main main.cpp Matrix.hpp)
add_executable(TestCV TestCV.cpp Matrix.hpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(TestCV ${OpenCV_LIBS})
