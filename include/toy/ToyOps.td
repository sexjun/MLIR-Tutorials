#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "toy/ToyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// mnemonic 指名字
class ToyOp<string mnemonic, list<Trait> traits = []> :
  Op<ToyDialect, mnemonic, traits>;

// Pure 是 Trait，表示没有 SideEffect 的纯函数
def AddOp : ToyOp<"add", [Pure]> {
  let summary = "add operation";
//   使用 Variadic<Type> 来描述可变参数： 使用 Option<Type> 来描述可选参数：
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
}

def SubOp : ToyOp<"sub", [Pure]> {
  let summary = "sub operation";
  //   在arguments字段里,添加Attribute属性
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
// 在这里添加需要验证的标记,然后在cpp文件中,实现该验证方法.
  let hasVerifier = true;
  // 新增构造函数
  let builders = [
    OpBuilder<
      (ins "mlir::Value":$lhs, "mlir::Value":$rhs),
      "build($_builder, $_state, lhs.getType(), lhs, rhs);"
    >
  ];
}

// [Pure] 在这里添加Trait
def ConstantOp : ToyOp<"const", [Pure]> {
  let summary = "const operation";
  let arguments = (ins APIntAttr:$value);
  let results = (outs AnyInteger:$result);
//   为OP自定义函数
  let extraClassDeclaration = [{
    int64_t getBitWidth() {
      return getResult().getType().getWidth();
    }
  }];
}

#endif
