add_executable(mlir-demo main.cpp)
target_link_libraries(
    mlir-demo
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    )
