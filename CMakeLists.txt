cmake_minimum_required(VERSION 2.8)

project(adapt-bw)

find_package(ImageMagick COMPONENTS Magick++ MagickCore REQUIRED)

include_directories(${ImageMagick_Magick++_INCLUDE_DIR})
include_directories("/usr/include/x86_64-linux-gnu/ImageMagick-6/")
add_definitions(-DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0)

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

add_executable(denoizer Denoizer.cpp)
target_link_libraries(denoizer ${ImageMagick_Magick++_LIBRARY} ${ImageMagick_MagickCore_LIBRARY} fftw3 fftw3_threads pthread)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	target_link_libraries(thresholder gomp)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
else ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	message(FATAL_ERROR "Unsupported compiler")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")	
