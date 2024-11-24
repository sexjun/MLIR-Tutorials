#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
using namespace std;
using namespace mlir;

int main(int argc, char ** argv) {

    // 1. 注册方言
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, mlir::arith::ArithDialect>();

    // 2. 读入文件
    auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);

    // 3. 简单输出 dialect
    cout << "从文件中读取mlir" << endl;
    src->print(llvm::outs());



    cout << "使用代码,生成mlir" << endl;
    // 4. create OpBuilder
    OpBuilder builder(&ctx);
    auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());

    // 5. 设置插入点
    builder.setInsertionPointToEnd(mod.getBody());

    // 6. 创建func
    auto i32 = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32, i32}, i32);
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "funName", funcType);

    // 7. 添加基本块
    auto entry = func.addEntryBlock();
    auto args = entry->getArguments();

    // 8. 设置插入点
    builder.setInsertionPointToEnd(entry);

    // 9. 创建arith.addi
    auto addi = builder.create<arith::AddIOp>(builder.getUnknownLoc(), args[0], args[1]);

    // 10. 创建func.return
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange({addi}));
    mod.dump();




    return 0;

}
