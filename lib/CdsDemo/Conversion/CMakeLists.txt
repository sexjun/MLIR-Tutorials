add_mlir_library(
    CdsPatterns
    Patterns.cpp
    DEPENDS
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRArithUtils
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefTransforms
    MLIRTensorDialect
    MLIRTensorUtils
)
