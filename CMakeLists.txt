cmake_minimum_required(VERSION 2.8)

project(adapt-bw)

find_package(ImageMagick COMPONENTS Magick++ REQUIRED)

include_directories(${ImageMagick_Magick++_INCLUDE_DIR})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set (CMAKE_CXX_FLAGS "/W3")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set (CMAKE_CXX_FLAGS "-Wall -fopenmp")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set (CMAKE_CXX_FLAGS "-Wall -ferror-limit=5")
else ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	message(FATAL_ERROR "Unsupported compiler")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")	

add_executable(thresholder Thresholder.cpp)
target_link_libraries(thresholder ${ImageMagick_Magick++_LIBRARY})

add_executable(rotator Rotator.cpp)
target_link_libraries(rotator ${ImageMagick_Magick++_LIBRARY})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	target_link_libraries(thresholder gomp)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
else ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	message(FATAL_ERROR "Unsupported compiler")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")	
