#include "PassDetail.h"
#include "Passes.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;
using namespace cf;

//===----------------------------------------------------------------------===//
// Target & Type Converter
//===----------------------------------------------------------------------===//

struct CFTarget : public ConversionTarget {
  CFTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Branches that have an index type are illegal
    addDynamicallyLegalOp<BranchOp>([](BranchOp op) {
      for (Type t : op.getOperandTypes())
        if (t.isIndex())
          return false;

      return true;
    });

    // Any other Op is legal
    markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  }
};

class Converter : public TypeConverter {
public:
  Converter() {
    addConversion([](Type type) { return type; });
    addConversion(convertIndexTypes);
  }

  static Optional<Type> convertIndexTypes(Type type) {
    if (IndexType indexType = type.dyn_cast<IndexType>()) {
      OpBuilder builder(type.getContext());

      // NOTE: Change to desired integer type
      // TODO: Add subflag instead of hardcoding
      return builder.getI64Type();
    }

    return llvm::None;
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Creates a cast from the index type to the desired integer type
arith::IndexCastOp createIndexCastOp(ConversionPatternRewriter &rewriter,
                                     Value val) {
  Location loc = val.getLoc();
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::IndexCastOp::getOperationName());

  arith::IndexCastOp::build(builder, state,
                            Converter().convertType(val.getType()), val);
  arith::IndexCastOp indexCastOp =
      cast<arith::IndexCastOp>(rewriter.create(state));

  return indexCastOp;
}

// Creates a cast from the integer type to the index type (reverse of the above)
arith::IndexCastOp createIndexCastOp(ConversionPatternRewriter &rewriter,
                                     BlockArgument bArg) {
  Location loc = bArg.getLoc();
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::IndexCastOp::getOperationName());

  arith::IndexCastOp::build(builder, state, builder.getIndexType(), bArg);
  arith::IndexCastOp indexCastOp =
      cast<arith::IndexCastOp>(rewriter.create(state));

  return indexCastOp;
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

class BranchPattern : public OpConversionPattern<BranchOp> {
public:
  using OpConversionPattern<BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Rewrite the branch operation with casted args
    rewriter.setInsertionPoint(op);
    std::vector<Value> newOperands = {};

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value operand = op.getOperand(i);

      if (operand.getType().isIndex()) {
        arith::IndexCastOp indexCastOp = createIndexCastOp(rewriter, operand);
        Value result = indexCastOp.getResult();
        newOperands.push_back(result);
      } else {
        newOperands.push_back(operand);
      }
    }

    Location loc = op.getLoc();
    OpBuilder builder(loc->getContext());
    OperationState state(loc, BranchOp::getOperationName());

    BranchOp::build(builder, state, newOperands, op.getDest());
    rewriter.create(state);
    rewriter.eraseOp(op);

    // Rewrite branch args and add casts
    rewriter.setInsertionPointToStart(op.getDest());

    for (unsigned i = 0; i < op.getDest()->getNumArguments(); ++i) {
      BlockArgument bArg = op.getDest()->getArgument(i);

      if (!bArg.getType().isIndex())
        continue;

      // NOTE: Might be better to use the rewriter
      bArg.setType(Converter().convertType(bArg.getType()));
      arith::IndexCastOp indexCastOp = createIndexCastOp(rewriter, bArg);

      // NOTE: Might be better to use the rewriter:
      // rewriter.replaceUsesOfBlockArgument(bArg, indexCastOp);
      // Failed to figure out how to erase the arguments without tripping a
      // "Can't erase a value that's in use" error
      bArg.replaceAllUsesExcept(indexCastOp, indexCastOp);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void populateCFIndexToIntConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctxt = patterns.getContext();

  patterns.add<BranchPattern>(ctxt);
}

namespace {
struct CFIndexToIntPass : public CFIndexToIntPassBase<CFIndexToIntPass> {
  void runOnOperation() override;
};
} // namespace

void CFIndexToIntPass::runOnOperation() {
  ModuleOp module = getOperation();

  CFTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateCFIndexToIntConversionPatterns(patterns);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

std::unique_ptr<Pass> cf::createCFIndexToIntPass() {
  return std::make_unique<CFIndexToIntPass>();
}
