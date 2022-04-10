cmake_minimum_required (VERSION 3.5)
project(exercises LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")

find_package(
OpenCV REQUIRED
)

include_directories( ${OpenCV_INCLUDE_DIRS} )

enable_language(CUDA)

add_executable(image_segmentation image_segmentation.cu)
target_link_libraries( image_segmentation ${OpenCV_LIBS} )

add_executable(image_segmentation_opt image_segmentation_opt.cu)
target_link_libraries( image_segmentation_opt ${OpenCV_LIBS} )
