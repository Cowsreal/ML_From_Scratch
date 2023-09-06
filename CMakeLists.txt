cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
project(ML_FROM_SCRATCH VERSION 0.1.0 LANGUAGES C CXX)
add_compile_options(-shared -std=c++11 -D EUCLID)

set(SOURCE_FILES_DATA_HANDLER
    ${CMAKE_CURRENT_SOURCE_DIR}/source/data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/data_handler.cpp
)

set(SOURCE_FILES_KNN
    ${CMAKE_CURRENT_SOURCE_DIR}/KNN/source/knn.cpp
)

add_library(DATA_HANDLER SHARED ${SOURCE_FILES_DATA_HANDLER})
target_include_directories(DATA_HANDLER PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

#add_library(KNN SHARED ${SOURCE_FILES_KNN})
add_executable(KNN ${SOURCE_FILES_KNN})
target_include_directories(KNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/KNN/include
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(KNN PRIVATE DATA_HANDLER)
