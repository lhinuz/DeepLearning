cmake_minimum_required (VERSION 3.2)

project (cameracalib)
set(CMAKE_CXX_STANDARD 11)

include_directories (${PROJECT_SOURCE_DIR}/../../opencv/output/include/opencv4 .)
link_directories (${PROJECT_SOURCE_DIR}/../../opencv/output/lib)

file (GLOB SOURCES src/*.cpp)
file (GLOB OPENCVLIBS ${PROJECT_SOURCE_DIR}/../../opencv/output/lib/*.dylib)

add_executable (cameracalib ${SOURCES})
target_link_libraries(cameracalib ${OPENCVLIBS})
