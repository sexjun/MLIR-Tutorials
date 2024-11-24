参考自官方文档

[Pass Infrastructure - MLIR](https://mlir.llvm.org/docs/PassManagement/#declarative-pass-specification)




# PASS的分类
pass分为二种

1. OP pass: 在某类型OP上运行的pass
2. interface pass: 在特定op interface上执行的pass

它们两个的区别是，Pass 里面 `getOperation()` 返回的是 Operation 还是 Interface。

## 定于与OP无关的pass

1. CRTP继承 `OperationPass`

2. 重写虚函数 `void runOnOperation()`

```C++
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct MyOperationPass : public PassWrapper<MyOperationPass, OperationPass<>> {
  void runOnOperation() override {
    // Get the current operation being operated on.
    Operation *op = getOperation();
    ...
  }
};
```



### 按照OP级别过滤的pass

- td文件

该pass只针对`FuncOP` 这个一个单独的op进行执行

```Shell
def AffineLoopUnrollAndJam : Pass<"affine-loop-unroll-jam", "func::FuncOp"> {
  let summary = "Unroll and jam affine loops";
  let constructor = "mlir::affine::createLoopUnrollAndJamPass()";
  let options = [
    Option<"unrollJamFactor", "unroll-jam-factor", "unsigned",
           /*default=*/"4",
           "Use this unroll jam factor for all loops (default 4)">,
  ];
}
```



- c++代码

```C++
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct MyFunctionPass : public PassWrapper<MyOperationPass, OperationPass<func::FuncOp>> {
  void runOnOperation() {
    // Get the current operation being operated on.
    func::FuncOp op = getOperation();
  }
};
```

### 按照接口进行过滤



```C++
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct MyFunctionPass : public PassWrapper<MyOperationPass, InterfacePass<FunctionOpInterface>> {
  void runOnOperation() {
    // Get the current operation being operated on.
    FunctionOpInterface op = getOperation();
  }
};
```





# td文件声明pass的语法


**函数声明：**

1. `MyPass` 是生成pass类的类名

2. `my-pass` 是用于 `opt` 工具执行时候，告诉ta我要执行这个pass的名字

3. `ModuleOp` 是说我们的pass 继承与 这个类。



```C++
def MyPass : Pass<"my-pass", "ModuleOp"> {
  let summary = "My Pass Summary";
  let description = [{
    Here we can now give a much larger description of `MyPass`, including all of
    its various constraints and behavior.
  }];

  // A constructor must be provided to specify how to create a default instance
  // of MyPass. It can be skipped for this specific example, because both the
  // constructor and the registration methods live in the same namespace.
  let constructor = "foo::createMyPass()";

  // Specify any options.
  let options = [
    Option<"option", "example-option", "bool", /*default=*/"true",
           "An example option">,
    ListOption<"listOption", "example-list", "int64_t",
               "An example list option">
  ];

  // Specify any statistics.
  let statistics = [
    Statistic<"statistic", "example-statistic", "An example statistic">
  ];
}
```



**有哪些**

1. summary， 简单摘要

2. description： 详细描述

3. dependentDialects： 依赖的属性，操作，类型

4. constructor： 构造函数

5. options： pass 选项列表

6. statistics： 通行证统计信息列表



# step by step write a pass

1. 创建CdsDemoPasses.td文件

```Shell
#ifndef CDS
#define CDS
include "mlir/Pass/PassBase.td"

def CdsDemoPass : Pass<"CdsDemoPass"> {
let summary = "My Pass Summary";
  let description = [{
    Here we can now give a much larger description of `MyPass`, including all of
    its various constraints and behavior.
  }];
}

#endif

```

1. 编写CmakeLists文件

```CMake
set(LLVM_TARGET_DEFINITIONS CdsDemoPasses.td)
mlir_tablegen(CdsDemoPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRCdsDemoPassesIncGen)
```

3.添加`CdsDemoPasses.h`文件

```C++
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

```

1. 实现`CdsDemoPasses.cpp`文件

```C++
#include "CdsDemo/CdsDemoPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::cdsdemo {

// pass的函数定义
#define GEN_PASS_DEF_CDSDEMOPASS
#include "CdsDemo/CdsDemoPasses.h.inc"

class CdsBasePass : public impl::CdsDemoPassBase<CdsBasePass> {
  using impl::CdsDemoPassBase<CdsBasePass>::CdsDemoPassBase;
  void runOnOperation() final {
    llvm::outs() << "pass name: " << getPassName() << "\n";

    auto ops = getOperation();

    ops->walk([](Operation *child) {
      llvm::outs() << "\t type:" << child->getName()
                   << "\t loc:" << child->getLoc() << "\n";
    });

    llvm::outs() << "====================================\n";
  }
};
} // namespace mlir::cdsdemo

```

1. 添加cpp文件的编译

```CMake
add_mlir_library(
    CdsPasses
    CdsDemoPasses.cpp
    DEPENDS
    MLIRCdsDemoPassesIncGen
)
```

6.在OPT中使用

```C++
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
using namespace mlir;
using namespace llvm;

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "my-mlir-pass"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

// 自己写的pass
#include "CdsDemo/CdsDemoPasses.h"

int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
//   mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  llvm::outs() << "running my mlir pass cds-demo\n";

  // 注册自己构造的Dialect
  //   registry
  //       .insert<func::FuncDialect, linalg::LinalgDialect,
  //       tosa::TosaDialect>();

  // 注册pass
  mlir::cdsdemo::registerPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt-cds", registry));
}

```







[Pattern Rewrite](Pass+1c8ef186-0002-4683-ad56-6ed1d0e17728/Pattern+Rewrite%20a8ea68ae-3b3d-44d0-bd3e-734339851651.md)

[](Pass+1c8ef186-0002-4683-ad56-6ed1d0e17728/%205c9235ed-8397-4dee-99ed-71562f42635e.md)



