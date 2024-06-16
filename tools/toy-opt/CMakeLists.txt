message("----------")
message("build tools")

add_mlir_tool(toy-opt2 toy-opt.cpp)
target_link_libraries(toy-opt2
    PRIVATE
    MLIRIR MLIRParser MLIRSupport
    Toy # 对应 #include "toy/ToyDialect.h"
    MLIROptLib # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
    MLIRFuncDialect # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
    MLIRTransforms # 对应 #include "mlir/Transforms/Passes.h"
)
