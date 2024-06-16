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
