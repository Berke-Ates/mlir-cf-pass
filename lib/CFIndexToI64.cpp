#include "PassDetail.h"
#include "Passes.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;
using namespace cf;

//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

struct CFTarget : public ConversionTarget {
  CFTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Any Op containing a basic block with an index argument is illegal
    markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void populateCFIndexToI64ConversionPatterns(RewritePatternSet &patterns) {
  // MLIRContext *ctxt = patterns.getContext();

  // TODO: Register Patterns here
}

namespace {
struct CFIndexToI64Pass : public CFIndexToI64PassBase<CFIndexToI64Pass> {
  void runOnOperation() override;
};
} // namespace

void CFIndexToI64Pass::runOnOperation() {
  ModuleOp module = getOperation();

  CFTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateCFIndexToI64ConversionPatterns(patterns);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

std::unique_ptr<Pass> cf::createCFIndexToI64Pass() {
  return std::make_unique<CFIndexToI64Pass>();
}
