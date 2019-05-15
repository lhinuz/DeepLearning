cmake_minimum_required (VERSION 3.2)

project (knn)
set(CMAKE_CXX_STANDARD 11)

include_directories (${PROJECT_SOURCE_DIR}/../../opencv/output/include/opencv4 .)
link_directories (${PROJECT_SOURCE_DIR}/../../opencv/output/lib)

file (GLOB SOURCES src/*.cpp)
file (GLOB OPENCVLIBS ${PROJECT_SOURCE_DIR}/../../opencv/output/lib/*.dylib)

add_executable (knn ${SOURCES})
target_link_libraries(knn ${OPENCVLIBS})
