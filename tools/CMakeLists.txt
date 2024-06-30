add_subdirectory(toy-opt)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})
add_subdirectory(include/toy)
add_subdirectory(lib)
add_subdirectory(tools/toy-opt)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
