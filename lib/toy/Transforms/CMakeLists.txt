add_mlir_library(
    ToyTransforms
    ConvertToyToArith.cpp
    DEPENDS
    MLIRToyTransformsIncGen
)
