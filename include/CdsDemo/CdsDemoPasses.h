#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {

namespace cdsdemo {

// pass声明
#define GEN_PASS_DECL
#include "CdsDemo/CdsDemoPasses.h.inc"

// pass注册函数
#define GEN_PASS_REGISTRATION
#include "CdsDemo/CdsDemoPasses.h.inc"

// void cds_pass_register() { impl::registerMyOperationPass(); }
} // namespace cdsdemo

} // namespace mlir
