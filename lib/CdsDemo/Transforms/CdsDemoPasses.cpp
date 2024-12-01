#include "CdsDemo/Transforms/CdsDemoPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::cdsdemo {

// pass的函数定义
#define GEN_PASS_DEF_CDSDEMOPASS
#include "CdsDemo/Transforms/CdsDemoPasses.h.inc"

class CdsBasePass : public impl::CdsDemoPassBase<CdsBasePass> {
  using impl::CdsDemoPassBase<CdsBasePass>::CdsDemoPassBase;
  void runOnOperation() final {
    llvm::outs() << "pass name: " << getPassName() << "\n";

    auto ops = getOperation();

    ops->walk([](Operation *child) {
      llvm::outs() << "\t type:" << child->getName()
                   << "\t loc:" << child->getLoc() << "\n";
    });

    llvm::outs() << "====================================\n";
  }
};
} // namespace mlir::cdsdemo
