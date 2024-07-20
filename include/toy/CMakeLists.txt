
# set(LLVM_TARGET_DEFINITIONS Toy.td)
# mlir_tablegen(Toy.h.inc -gen-op-decls)
# mlir_tablegen(Toy.cpp.inc -gen-op-defs)
# mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)
# add_public_tablegen_target(MLIRToyIncGen)

# dialect
add_mlir_dialect(Toy toy)
add_mlir_doc(FooOps FooDialect Dialects/ -gen-dialect-doc)

# pass
set(LLVM_TARGET_DEFINITIONS ToyPasses.td)
mlir_tablegen(ToyPasses.h.inc -gen-pass-decls)
message("MLIRToyTransformsIncGen: gen---------------->")
add_public_tablegen_target(MLIRToyTransformsIncGen)

# interface
# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ToyInterfaces.td)
mlir_tablegen(ToyInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ToyInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyInterfacesIncGen)
