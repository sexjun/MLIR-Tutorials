
# op
# add_mlir_dialect(StandaloneOps standalone)
# add_mlir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)
# add_mlir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)

# pass
set(LLVM_TARGET_DEFINITIONS CdsDemoPasses.td)
mlir_tablegen(CdsDemoPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRCdsDemoPassesIncGen)

# interface
# Most dialects should use add_mlir_interfaces().
# set(LLVM_TARGET_DEFINITIONS ToyInterfaces.td)
# mlir_tablegen(ToyInterfaces.h.inc -gen-op-interface-decls)
# mlir_tablegen(ToyInterfaces.cpp.inc -gen-op-interface-defs)
# add_public_tablegen_target(ToyInterfacesIncGen)
