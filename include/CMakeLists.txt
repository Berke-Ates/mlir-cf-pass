set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CFIndexToI64)
add_public_tablegen_target(MLIRCFIndexToI64PassIncGen)
