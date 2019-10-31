cmake_minimum_required(VERSION 3.1.0)

project(Analysis)

# Save the command line compile commands in the build output
#set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "\n\n Opencv2 is missing") 
endif()

include_directories(${OpenCV_INCLUDE_DIRS} .)

add_executable(Analysis Analysis.cpp opticalflow.cpp)
set_property(TARGET Analysis PROPERTY CXX_STANDARD 11)
target_link_libraries(Analysis ${OpenCV_LIBS})
#include_directories(Analysis)
#set_target_properties (Analysis PROPERTIES FOLDER "Examples")

