#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {

namespace cdsdemo {

#define GEN_PASS_DECL
#include "CdsDemo/CdsPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "CdsDemo/CdsPasses.h.inc"

// void cds_pass_register() { impl::registerMyOperationPass(); }
} // namespace cdsdemo

} // namespace mlir
