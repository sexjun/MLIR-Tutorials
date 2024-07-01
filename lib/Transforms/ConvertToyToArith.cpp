#define GEN_PASS_DEF_CONVERTTOYTOARITH
#include "toy/ToyPasses.h"
#include "llvm/Support/raw_ostream.h"

struct ConvertToyToArithPass :
    toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>
{
  // 使用父类的构造函数
  using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;
  void runOnOperation() final {
    getOperation()->print(llvm::errs());
  }
};

std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass() {
  return std::make_unique<ConvertToyToArithPass>();
}
