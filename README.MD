# 初始化

- read mlir from file
```c++
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>


using namespace std;
using namespace mlir;

int main(int argc, char ** argv) {

    // 1. 注册方言
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, mlir::arith::ArithDialect>();

    // 2. 读入文件
    auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);

    // 3. 简单输出 dialect
    src->print(llvm::outs());

    cout << "-----------------> " << endl;
    src->dump();



    return 0;

}

```

- execute

```shell
(base) ➜  mlir_cds git:(master) ✗ /Users/chendongsheng/github/mlir_cds/build/mlir-toy /Users/chendongsheng/github/mlir_cds/test.mlir
module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
----------------->
module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
```
