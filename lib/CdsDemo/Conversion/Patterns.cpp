#include "CdsDemo/Conversion/Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::cdsdmeo {

struct ConversionAddToMul final : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (addOp->hasSuccessors()) {
      rewriter.replaceOp(addOp, tosa::MulOp());
      return success();
    } else {
      return llvm::failure();
    }
  }
};

void register_cdsdemo_patterns(MLIRContext *context,
                               RewritePatternSet &patterns) {
  patterns.insert<ConversionAddToMul>(context);
};

} // namespace mlir::cdsdmeo
