using namespace mlir;
using namespace cf;

//===----------------------------------------------------------------------===//
// Target & Type Converter
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void populateCFIndexToI64ConversionPatterns(RewritePatternSet &patterns,
                                            TypeConverter &converter) {
  MLIRContext *ctxt = patterns.getContext();

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
  MemrefToMemletConverter converter;

  RewritePatternSet patterns(&getContext());
  populateCFIndexToI64ConversionPatterns(patterns, converter);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

std::unique_ptr<Pass> conversion::createCFIndexToI64Pass() {
  return std::make_unique<CFIndexToI64Pass>();
}
