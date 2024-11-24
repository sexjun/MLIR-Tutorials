#include "CdsDemo/CdsDemoPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::cdsdemo {

#define GEN_PASS_DEF_CDSDEMOPASS
#include "CdsDemo/CdsDemoPasses.h.inc"

class CdsBasePass : public impl::CdsDemoPassBase<CdsBasePass> {
  using impl::CdsDemoPassBase<CdsBasePass>::CdsDemoPassBase;
  void runOnOperation() final {
    llvm::errs() << "get name: " << "\n";
    llvm::outs() << "----------> ConvertToyToArithPass run" << "\n";
    llvm::outs() << "here is the pass of cdsbasepass" << "\n";
  }
};
} // namespace mlir::cdsdemo
