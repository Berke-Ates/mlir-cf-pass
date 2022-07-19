#ifndef CF_PASSES_TD
#define CF_PASSES_TD

include "mlir/Pass/PassBase.td"

def CFIndexToIntPass : Pass<"cf-index-to-int", "ModuleOp"> {
  let summary = "Convert index types in cf branches to int";
  let constructor = "mlir::cf::createCFIndexToIntPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

#endif // CF_PASSES_TD
