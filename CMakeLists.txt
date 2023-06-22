cmake_minimum_required(VERSION 3.1)
##cd build
##cmake ..
##make install

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#add_definitions(-DEIGEN_USE_LAPACKE)
project(gsadf VERSION 0.1.0)


set(CMAKE_CXX_FLAGS "-Wall -O3 -m64 -fopenmp  -mavx -mfma  -march=native ")
#-m64 -fopenmp -march=native  -mavx2 -DNDEBUG

# Include dir
include_directories(/usr/local/include)
include_directories(/usr/include/eigen3)
include_directories(/usr/include/eigen3/unsupported)
include_directories(/home/rstudio/gsadf_CPP/libcxxwrap-julia/include)
include_directories(/opt/julia-1.9.0/include/julia)

set(LIB_OUT "lib")

# Source files

aux_source_directory(. DIR_SRCS)



add_library(gsadf SHARED ${DIR_SRCS})


set_target_properties(gsadf PROPERTIES PREFIX "")

set_target_properties(gsadf PROPERTIES OUTPUT_NAME "gsadf")

install(TARGETS gsadf DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/./)