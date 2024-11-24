message("----------")
message("build tools for toy-opt")
message("----------")

add_mlir_tool(toy-opt-cds toy-opt.cpp)
target_link_libraries(toy-opt-cds
    PRIVATE
    MLIRIR MLIRParser MLIRSupport

    Toy # 对应 #include "toy/ToyDialect.h"
    CdsPasses

    MLIROptLib # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
    MLIRFuncDialect # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
    MLIRTransforms # 对应 #include "mlir/Transforms/Passes.h"

    MLIRLinalgDialect # add linalg dialect
    ToyTransforms
)
