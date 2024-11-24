
# MLIR-TutoRials

MLIR的学习项目，通过该项目来了解MLIR的各个功能组件。

- 文档
    - [如何搭建MLIR项目的环境](https://github.com/sexjun/MLIR-Tutorials/blob/master/doc/%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md)
    - [MLIR的基本用法（了解即可）](https://github.com/sexjun/MLIR-Tutorials/blob/master/doc/MLIR%E7%9A%84%E5%9F%BA%E7%A1%80%E7%94%A8%E6%B3%95.md)
    - [通过ODS创建pass](https://github.com/sexjun/MLIR-Tutorials/blob/master/doc/%E5%A6%82%E4%BD%95%E9%80%9A%E8%BF%87td%E6%96%87%E4%BB%B6%E5%88%9B%E5%BB%BApass.md)




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

