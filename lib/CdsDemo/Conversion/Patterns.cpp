#include "CdsDemo/Conversion/Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>

namespace mlir::cdsdmeo {

struct ConversionReduceMinToReduceMax final
    : public OpRewritePattern<tosa::ReduceMinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReduceMinOp reduceMinOp,
                                PatternRewriter &rewriter) const override {
    // 获取reducemin操作的输入张量
    Value input = reduceMinOp.getInput();

    // 获取是否保持维度的属性，通过检查操作的属性字典
    // 创建一个新的tosa方言下的ReduceMaxOp，参数与原ReduceMinOp保持一致
    auto reduceMaxOp = rewriter.create<tosa::ReduceMaxOp>(
        reduceMinOp.getLoc(), input, reduceMinOp.getAxis());

    // 用新创建的ReduceMaxOp替换原有的ReduceMinOp
    rewriter.replaceOp(reduceMinOp, reduceMaxOp.getResult());

    return success();
  }
};

void register_cdsdemo_patterns(MLIRContext *context,
                               RewritePatternSet &patterns) {
  patterns.insert<ConversionReduceMinToReduceMax>(context);
};

} // namespace mlir::cdsdmeo
