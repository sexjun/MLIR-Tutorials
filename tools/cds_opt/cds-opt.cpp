#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
using namespace mlir;
using namespace llvm;

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "my-mlir-pass"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

// 自己写的pass
#include "CdsDemo/Transforms/CdsDemoPasses.h"

#include "CdsDemo/Conversion/Patterns.h"

int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
//   mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  llvm::outs() << "running my mlir pass cds-demo\n";

  // 注册自己构造的Dialect
  //   registry
  //       .insert<func::FuncDialect, linalg::LinalgDialect,
  //       tosa::TosaDialect>();

  // 注册pass
  mlir::cdsdemo::registerPasses();
  //   register_cdsdemo_patterns()
  return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt-cds", registry));
}
