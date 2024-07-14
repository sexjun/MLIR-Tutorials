#ifndef TOY_PASSES_TD
#define TOY_PASSES_TD

include "mlir/Pass/PassBase.td"


// 会生成一个 ConvertToyToArithBase的类, 需要继承该类完成pass编写.
// 执行该pass的方法: ./toy-opt-cds --convert-toy-to-arith /Users/chendongsheng/github/mlir_cds/cases/convert.mlir
def ConvertToyToArith : Pass<"convert-toy-to-arith"> {
  let summary = "Convert Toy To Arith";
  let constructor = "toy::createConvertToyToArithPass()";
  let options = [
    Option<"name", "name", "std::string", "", "help">
  ];
}
#endif
