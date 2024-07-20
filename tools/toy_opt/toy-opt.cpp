#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
// 导入我们新建的 Dialect
#include "toy/ToyDialect.h"
// import new pass
#include "toy/ToyPasses.h"
using namespace mlir;
using namespace llvm;

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "my-mlir-pass"

int main(int argc, char ** argv) {
  DialectRegistry registry;

    LLVM_DEBUG(llvm::dbgs() << "Running my MLIR pass\n");

  // 注册 Dialect
  registry.insert<toy::ToyDialect, func::FuncDialect>();
  // 注册两个 Pass
  registerCSEPass();
  registerCanonicalizerPass();
  // register pass
  toy::registerPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt-cds", registry));
}
