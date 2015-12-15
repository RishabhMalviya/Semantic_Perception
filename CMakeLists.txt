project(Semantic_Perception)
cmake_minimum_required(VERSION 2.8)


#LIBRARIES FRO LINKING
#Vigra
include_directories(
    include
)
file(GLOB_RECURSE VigraImpex include/impex/*)
add_library(VigraImpex ${VigraImpex})

#mlpack
find_library(MLPack NAMES libmlpack.so PATHS /usr/local/lib)

#LibXml2
find_package(LibXml2 REQUIRED)

#Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

#Boost
find_package(Boost 1.54.0 REQUIRED)



#EXECUTABLES
add_executable(features_slic src/features_slic.cpp)
target_link_libraries(features_slic
    VigraImpex
    ${MLPack}
)

add_executable(GT_SLICLabels_extraction src/GT_SLICLabels_extraction.cpp)
target_link_libraries(GT_SLICLabels_extraction
    VigraImpex
    ${MLPack}
)

add_executable(kNN src/kNN.cpp)
target_link_libraries(kNN
    VigraImpex
    ${MLPack}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)

