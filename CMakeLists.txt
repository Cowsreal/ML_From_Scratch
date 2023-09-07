cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
project(ML_FROM_SCRATCH VERSION 0.1.0 LANGUAGES C CXX)
add_compile_options(-shared -std=c++11 -DMANHATTAN -DKMEANS)

set(SOURCE_FILES_MAIN
    ${CMAKE_CURRENT_SOURCE_DIR}/MAIN/source/main.cpp
)

set(SOURCE_FILES_DATA_HANDLER
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA_HANDLER/source/data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA_HANDLER/source/data_handler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DATA_HANDLER/source/common.cpp
)

set(SOURCE_FILES_KNN
    ${CMAKE_CURRENT_SOURCE_DIR}/KNN/source/knn.cpp
)

set(SOURCE_FILES_KMEANS
    ${CMAKE_CURRENT_SOURCE_DIR}/KMEANS/source/kmeans.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KMEANS/source/cluster.cpp
)

add_library(DATA_HANDLER SHARED ${SOURCE_FILES_DATA_HANDLER})
target_include_directories(DATA_HANDLER PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/DATA_HANDLER/include)

add_library(KNN SHARED ${SOURCE_FILES_KNN})
target_include_directories(KNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/KNN/include
                                        ${CMAKE_CURRENT_SOURCE_DIR}/DATA_HANDLER/include)

add_library(KMEANS SHARED ${SOURCE_FILES_KMEANS})
target_include_directories(KMEANS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/KMEANS/include
                                        ${CMAKE_CURRENT_SOURCE_DIR}/DATA_HANDLER/include)

add_executable(MAIN ${SOURCE_FILES_MAIN})
target_include_directories(MAIN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/MAIN/include
                                        ${CMAKE_CURRENT_SOURCE_DIR}/DATA_HANDLER/include
                                        ${CMAKE_CURRENT_SOURCE_DIR}/KNN/include
                                        ${CMAKE_CURRENT_SOURCE_DIR}/KMEANS/include)


target_link_libraries(KNN PRIVATE DATA_HANDLER)
target_link_libraries(KMEANS PRIVATE DATA_HANDLER)
target_link_libraries(MAIN PRIVATE DATA_HANDLER)
target_link_libraries(MAIN PRIVATE KNN)
target_link_libraries(MAIN PRIVATE KMEANS)