cmake_minimum_required(VERSION 2.8)

project(adapt-bw)

find_package(ImageMagick COMPONENTS Magick++ REQUIRED)

include_directories(${ImageMagick_Magick++_INCLUDE_DIR})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set (CMAKE_CXX_FLAGS "/W3")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set (CMAKE_CXX_FLAGS "-Wall -fopenmp")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set (CMAKE_CXX_FLAGS "-Wall -ferror-limit=5 -fopenmp")
	include_directories("/opt/intel/omp/include")
	link_directories("/opt/intel/omp/lib")
else ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	message(FATAL_ERROR "Unsupported compiler")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")	

add_executable(main Main.cpp)
target_link_libraries(main ${ImageMagick_Magick++_LIBRARY})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	target_link_libraries(main gomp)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	target_link_libraries(main iomp5)
else ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	message(FATAL_ERROR "Unsupported compiler")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")	
