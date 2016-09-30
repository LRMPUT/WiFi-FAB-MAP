# cmake requirements
cmake_minimum_required(VERSION 2.8)

# Build options have to be before PROJECT(...)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE PATH "Configuration types")
set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Current build configuration")

# PUTSLAM Project configuration
project(WiFiFABMAP)
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread -g")

# Include directory
include_directories("${CMAKE_SOURCE_DIR}/include")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

set(FABMAP_SOURCES
	src/BOWMSCTrainer.cpp
	src/ChowLiuTree.cpp
	src/FabMap.cpp)
	
add_library(fabmap
			${FABMAP_SOURCES})
			

set(WiFiFABMAP_SOURCES
	src/main.cpp)
add_executable(wififabmap
				${WiFiFABMAP_SOURCES})
target_link_libraries(wififabmap
					fabmap
					${OpenCV_LIBS})
							