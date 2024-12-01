#ifndef CDS_PATTHERNS
#define CDS_PATTHERNS

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir::cdsdemo {
void register_cdsdemo_patterns(MLIRContext *context,
                               RewritePatternSet &patterns);
}

#endif
