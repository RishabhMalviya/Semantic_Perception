project(Semantic_Perception)
cmake_minimum_required(VERSION 2.8)



#Vigra Headers
include_directories(
    include
)
file(GLOB_RECURSE VigraImpex include/impex/*)
add_library(VigraImpex ${VigraImpex})

#mlpack Headers
#add_library(MLPack SHARED IMPORTED /usr/loca/lib/libmlpack.so)
find_library(MLPack NAMES libmlpack.so PATHS /usr/local/lib)

#LibXml2 Headers
find_package(LibXml2 REQUIRED)

#Armadillo Headedrs
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#Boost Headers
find_package(Boost 1.54.0 REQUIRED)



add_executable(features_slic src/features_slic.cpp)
target_link_libraries(features_slic
    VigraImpex
    ${MLPack}
)

#add_executable(arma_test src/arma_test.cpp)
#target_link_libraries(arma_test
#    ${ARMADILLO_LIBRARIES}
#    ${MLPack}
#)


#add_executable(io_test src/io_test.cpp)
#target_link_libraries(io_test
#    VigraImpex
#)

##ONLY SEGMENTATION
#add_executable(watershed_test src/watershed_test.cpp)
#target_link_libraries(watershed_test
#    VigraImpex
#)


add_executable(slic_test src/slic_test.cpp)
target_link_libraries(slic_test
    VigraImpex
    ${MLPack}
)

#add_executable(seeded_test src/seeded_test.cpp)
#target_link_libraries(seeded_test
#    VigraImpex
#)

#ONLY FEATURE EXTRACTION


#COMBINED
#add_executable(features_slic src/features_slic.cpp)
#target_link_libraries(features_slic
#    VigraImpex
#)
