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

## 5. 基本的dialect工程



[cmkae mlir结合使用指南](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)

对于 方言 + OP

```cmake
add_mlir_dialect(Toy toy)

1. 新建一个文件
2. 包含方言和OP的定义
## file name: Toy.td

#ifndef TOY_TD
#define TOY_TD
include "toy/ToyDialect.td"
include "toy/ToyOps.td"

#endif


# set(LLVM_TARGET_DEFINITIONS Toy.td)
# mlir_tablegen(Toy.h.inc -gen-op-decls)
# mlir_tablegen(Toy.cpp.inc -gen-op-defs)
# mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)
# add_public_tablegen_target(MLIRToyIncGen)

add_mlir_dialect(Toy toy)

```







1. ToyDialect.td 定义Dialect的名字和cpp命名空间
2. ToyOps.td 定义OP
3. Toy.td 将 1 和 2 连接在一起,用于 `tablegen` 生成
4. 用于 `cmake` 生成
```cmake
# Toy 是方言名字
# toy是指toy.td
# 该定义会生成一个名为 MLIRxxxIncGen的cmake target
add_mlir_dialect(Toy toy)
```
5. `build/include/toy` 生成在这里,
    1. ToyDialect.{h,cpp}.inc：存 Dialect 的定义和实现
    2. Toy.{h,cpp}.inc：存 Op 的定义和实现
6. `ToyDialect.h` 加载 `Dialect` 的定义
7. `ToyOps.h` 加载 `OP` 的定义
8. `toy.cpp` 加载 `Dialect` 和 `OP`
9. 编译刚刚所有的代码
```cmake

add_mlir_library(Toy toy.cpp DEPENDS MLIRToyIncGen)
```



这样完成后就可以通过:  `cse` 和 `canonicalize` 了. 为什么呢? 因为我们给OP 定义了`PURE`属性.

```c++

./toy-opt-cds -cse  /Users/chendongsheng/github/mlir_cds/cases/cse.mlir
./toy-opt-cds -canonicalize  /Users/chendongsheng/github/mlir_cds/cases/cse.mlir
```

### 5.1 OP verify

定义 OP 的时候,加上`let hasVerifier = true;` 属性,则可以在`toy.cpp`文件中添加;

```c++
using namespace mlir;
LogicalResult SubOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return this->emitError() << "Lhs Type " << getLhs().getType()
      << " not equal to rhs " << getRhs().getType();
  return success();
}
```

进行OP的验证.

### 5.2 TableGen OP 详细介绍

- `ToyOps.td` : OP定义文件



1. 声明一个`ToyOp`, 继承父`Op`对象.

```c++
// mnemonic 指名字, list<Trait> 是指有一系列特性
class ToyOp<string mnemonic, list<Trait> traits = []> :
  Op<ToyDialect, mnemonic, traits>;
```

2. 定义OP

   1. `add` 是这个OP的名字

   2. `Constraint` :  约束是指属性和数据类型的约束.

      1. `Attribute`: 常见的属性   `mlir/IR/CommonAttrConstraints.td`
      2. `Type`: 内置的数据类型

   3. `Verify`

      1. 在OP声明的时候加上 `let hasVerifier=true`

      2. 在`toy.cpp`中实现该 `verifier`

         1. ```c++
            using namespace mlir;
            LogicalResult SubOp::verify() {
              if (getLhs().getType() != getRhs().getType())
                return this->emitError() << "Lhs Type " << getLhs().getType()
                  << " not equal to rhs " << getRhs().getType();
              return success();
            }
            ```

   4. Builder: 自定义 构造函数

      1. ```c++
         let builders = [
             OpBuilder<
               (ins "mlir::Value":$lhs, "mlir::Value":$rhs),
               "build($_builder, $_state, lhs.getType(), lhs, rhs);"
             >
           ];
         ```

      2. 首先，mlir 会自动为我们生成 `build($_builder, $_state, ResultType, LhsValue, RhsValue)` 的 builder

      3. 我们的 builder 通过 `lhs.getType()` 推断 result 的类型，并调用 mlir 生成好的 builder，实现自动推断类型

```c++
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

  // 自定义函数
  let extraClassDeclaration = [{
    int64_t getBitWidth() {
      return getResult().getType().getWidth();
    }
  }];
}
```



## 6. pass

pass分为二种

1. OP pass: 在某类型OP上运行的pass
2. interface pass: 在特定op interface上执行的pass

它们两个的区别是，Pass 里面 `getOperation()` 返回的是 Operation 还是 Interface。

### 6.1. OP pass

1. **CRTP**继承`OperationPass`
2. 重写虚函数 `void runOnOperation()`

- 与OP无关的pass (所有op都执行)

```c++
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

- 带有过滤的针对OP执行的pass

```c++
struct MyFunctionPass : public PassWrapper<MyOperationPass, OperationPass<>> {
  /// This method is used to provide additional static filtering, and returns if the
  /// pass may be scheduled on the given operation type.
  bool canScheduleOn(RegisteredOperationName opInfo) const override {
    return opInfo.hasInterface<FunctionOpInterface>();
  }

  void runOnOperation() {
    // Here we can freely cast to FunctionOpInterface, because our `canScheduleOn` ensures
    // that our pass is only executed on operations implementing that interface.
    FunctionOpInterface op = cast<FunctionOpInterface>(getOperation());
  }
};
```

- 基于过滤OP执行pass的基础,提供语法糖,仅针对某一类型OP执行的pass

```c++
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

- 按照接口进行过滤

```c++
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

### 6.2 pass manager

1. `Passmanager`       `OppassManager`



```c++
// Create a top-level `PassManager` class.
auto pm = PassManager::on<ModuleOp>(ctx);

// Add a pass on the top-level module operation.
pm.addPass(std::make_unique<MyModulePass>());


void registerMyPass() {
  PassRegistration<MyPass>();
}
```







tablegen 来定义 Pass

1. 创建ToyPasses.td
2. cmake配置生成代码
    ```cmake
    set(LLVM_TARGET_DEFINITIONS ToyPasses.td)
    mlir_tablegen(ToyPasses.h.inc -gen-pass-decls)
    add_public_tablegen_target(MLIRToyTransformsIncGen)
    ```
3. 声明pass的头文件 ToyPasses.h：Pass 的头文件
4. 声明pass的实现 Transforms/ConvertToyToArith.cpp：一个基本实现
5. cmake添加上library
```cmake
add_mlir_library(
  ToyTransforms
  ConvertToyToArith.cpp
  DEPENDS MLIRToyTransformsIncGen
)
```
6. 注册pass
```c++
toy::registerPasses();
return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
```

定义pass的执行
1. pass在指定op上执行
2. pass在指定interface上执行



## 7. pattern rewrite (toy-3)

ODS有3种:

1. 直接在`td` 文件中实现

2. 在`td` 文件中声明, 然后`c++` 函数做具体实现.

3. 给`op`声明 `let hasCanonicalizeMethod = 1;`属性,然后在

   1. `LogicalResult circt::MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) ` 函数中实现.

4. 使用`RwritePattern`

   ```c++
   struct ConvertTFLeakyRelu : public RewritePattern {
     ConvertTFLeakyRelu(MLIRContext *context)
         : RewritePattern("tf.LeakyRelu", 1, context) {}
   
     LogicalResult match(Operation *op) const override {
       return success();
     }
   
     void rewrite(Operation *op, PatternRewriter &rewriter) const override {
       rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
           op, op->getResult(0).getType(), op->getOperand(0),
           /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
     }
   };
   ```









## 7.3 PDL

> https://mlir.llvm.org/docs/PDLL/#

- 已经有pattern rewrite了,为什么还要PDL呢?

​	PDL 语言 (PDLL) 提供了一种从头开始设计的声明性模式语言，用于表示 MLIR 模式重写。 PDLL 旨在通过直观的界面原生支持在所有 MLIR 构造上编写匹配器，该界面可用于提前 (AOT) 和即时 (JIT) 模式编译。

​	Tablegen DRR (TDRR)，即表驱动的声明性重写规则，是一种声明性 DSL,TableGen 语言不像 LLVM 那样适合 MLIR 模式的结构。

​	TDRR 的问题很大程度上源于使用 TableGen 作为 DSL 的宿主语言。这些问题的根源在于 TableGen 的结构与 MLIR 的结构不匹配，以及 TableGen 与 MLIR 的动机目标不同。





## 8. 如何自定义type

