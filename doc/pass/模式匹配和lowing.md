Transform： 方言内转化
Conversion: 方言间转化
Analysis： 分析pass

## Pattern （Conversion）
`Pattern` 主要是 `Conversion` 任务中， 用于方言间的转化。
`Pattern`的注册并不是在`Conversion`中完成，而是在`Transform`中的 `Passes.td` 文件中，声明一个新的pass， 在这个新pass中完成对一系列 Pattern 的注册

## Transform
`pass` 其实在IREE项目可以发现
主要是在 `Transform` 任务中，用与方言内部的转化。 pass的声明都是在 `Passes.td`文件中完成的。




## Analysis
与Transform有关的一个概念是 `Analysis` ,概念上与 Transform 类似，不同的是 Analysis 计算相关信息，但是不修改图。
在MLIR中， analysis 不是pass， 是单独的类，这些类需要延迟计算并缓存，以避免不必要的从新计算，MLIR 中的 analysis 需要符合以下要求

1. 构造函数入参是`Operation*` 或者 `Operation*` 和 `AnalysisManager &`
   1. 提供的` AnalysisManager & `应该用于查询任何必要的分析依赖项。
2. 必须不能修改op


比如说有一个类是：
```c++
/// An interesting analysis.
struct MyOperationAnalysis {
  // Compute this analysis with the provided operation.
  MyOperationAnalysis(Operation *op);
};

```

然后在 Transform 的pass中，就可以通过
```c++

void MyOperationPass::runOnOperation() {
  // Query MyOperationAnalysis for the current operation.
  // 得到一个Analysis的对象
  MyOperationAnalysis &myAnalysis = getAnalysis<MyOperationAnalysis>();

  // 或者试图获取一个已经缓存的对象
  auto optionalAnalysis = getCachedAnalysis<MyOperationAnalysis>();
  if (optionalAnalysis)
}
```


<!-- `OpConversionPattern` -->


还需要了解 `Analysis` 和 `TransformExtensions` 的作用， 以及pass 的 `pipleline`


## pipleline

是说将一堆指定的pass 组成的pass集合（成为pipleline），然后逐个运行。

```c++
void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<MyPass>());
  pm.addPass(std::make_unique<MyOtherPass>());
}

void registerMyPasses() {
  // Register an existing pipeline builder function.
  PassPipelineRegistration<>(
    "argument", "description", pipelineBuilder);

  // Register an inline pipeline builder.
  PassPipelineRegistration<>(
    "argument", "description", [](OpPassManager &pm) {
      pm.addPass(std::make_unique<MyPass>());
      pm.addPass(std::make_unique<MyOtherPass>());
    });
}
```




```c++
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

// 主函数，用于构建和运行 Pass Pipeline
int main(int argc, char **argv) {
    // 初始化 LLVM 库
    llvm::InitLLVM y(argc, argv);
    // 创建 MLIR 上下文
    MLIRContext context;

    // 创建一个模块级别的 Pass Manager
    OpPassManager pm(ModuleOp::getOperationName());

    // 添加特定的 Pass 到 Pass Manager
    // 添加内联 Pass，将函数调用替换为函数体
    pm.addNestedPass<func::FuncOp>(createInlinerPass());
    // 添加常量折叠 Pass，对常量表达式进行求值
    pm.addNestedPass<func::FuncOp>(createConstantFoldPass());
    // 添加死代码消除 Pass，移除未使用的代码
    pm.addNestedPass<func::FuncOp>(createDeadCodeEliminationPass());

    // 创建一个源管理器，用于管理输入文件
    llvm::SourceMgr sourceMgr;
    // 从标准输入读取 MLIR 代码
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getSTDIN(), llvm::SMLoc());

    // 解析输入的 MLIR 代码，创建模块
    OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, &context));
    if (!module) {
        return 1;
    }

    // 运行 Pass Pipeline
    if (failed(pm.run(*module))) {
        llvm::errs() << "Failed to run pass pipeline\n";
        return 1;
    }

    // 打印转换后的 MLIR 模块到标准输出
    module->print(llvm::outs());
    return 0;
}
```
