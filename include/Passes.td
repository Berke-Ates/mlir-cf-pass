#ifndef CF_PASSES_TD
#define CF_PASSES_TD

include "mlir/Pass/PassBase.td"

def CFIndexToI64Pass : Pass<"cf-index-to-i64", "ModuleOp"> {
  let summary = "Convert index types in cf branches to i64";
  let constructor = "mlir::cf::createCFIndexToI64Pass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

#endif // CF_PASSES_TD
