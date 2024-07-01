add_mlir_dialect(Toy toy)
add_mlir_doc(FooOps FooDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ToyPasses.td)
mlir_tablegen(ToyPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRToyTransformsIncGen)


add_custom_target(header-2 DEPENDS MLIRToyIncGen MLIRToyTransformsIncGen)
