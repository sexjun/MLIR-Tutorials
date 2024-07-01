#ifndef TOY_PASSES_TD
#define TOY_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertToyToArith : Pass<"convert-toy-to-arith"> {
    // 简短单行摘要
  let summary = "Convert Toy To Arith";
// 创建传递的默认实例的代码块。
  let constructor = "toy::createConvertToyToArithPass()";
  let options = [
    Option<"name", "name", "std::string", "", "help">
  ];
}

def DCE : Pass<"toy-dce", "toy::FuncOp"> {
  let summary = "dce";
  let constructor = "toy::createDCEPass()";
}
#endif
