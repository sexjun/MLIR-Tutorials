#include "mlir/IR/Operation.h"
#include "llvm-c/Analysis.h"

/// An interesting analysis.
struct MyOperationAnalysis {
  // Compute this analysis with the provided operation.
  MyOperationAnalysis(mlir::Operation *op) {
    (void)op;
    return;
  }
};
