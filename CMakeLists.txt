cmake_minimum_required(VERSION 3.28.0)
project(cds_mlir VERSION 0.1.0 LANGUAGES C CXX)

cmake_policy(SET CMP0116 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json 便于代码高亮
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_INSTALL_PREFIX "/usr/local/cds_bin")

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 设置生成的dialect的头文件路径
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


add_executable(mlir-toy main.cpp)
target_link_libraries(
    mlir-toy
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    )


add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/toy-opt)
add_custom_target(header ALL DEPENDS MLIRToyIncGen)
