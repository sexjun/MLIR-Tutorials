# 初始化

## 1. 从mlir文件中读入
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

## mlir op的结构

通过代码来生成一个mlir op,体会一下mlir op的结构

```mlir
module {
  func.func @funName(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

OP 是由以下四部分组成的.
- Operand：这个 Op 接受的操作数 (输入参数)
- Result：这个 Op 生成的新 Value (返回值)
- Attribute：可以理解为编译器常量 (常量)
- Region：这个 Op 内部的 Region (region)


## 3. mlir的类型转化

1. OP 类型的转化
所有op的父类都是 `Opration`

```c++
using namespace llvm;
void myCast(Operation * op) {
  auto res = cast<AddOp>(op); // 直接转换，失败报错
  auto res = dyn_cast<AddOp>(op); // 尝试转换，失败返回 null，op为null时报错
  auto res = dyn_cast_if_present<AddOp>(op); // 类似 dyn_cast，op为null时返回null
}
```

2. `type` and `Attribute` convert

same with op conversion.

![alt text](image-1.png)

## 4. mlir图结构

## 5. 
