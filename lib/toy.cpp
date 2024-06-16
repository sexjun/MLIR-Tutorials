#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"
#include "toy/ToyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "toy/Toy.cpp.inc"
using namespace toy;

// 把默认 Dialect 和 Op 的默认实现加载进来
void ToyDialect::initialize() {
  // 下面的代码会生成 Op 的列表，专门用来初始化
  addOperations<
#define GET_OP_LIST
#include "toy/Toy.cpp.inc"
  >();
}

using namespace mlir;
LogicalResult SubOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return this->emitError() << "Lhs Type " << getLhs().getType()
      << " not equal to rhs " << getRhs().getType();
  return success();
}
