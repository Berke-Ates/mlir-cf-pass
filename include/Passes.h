#ifndef CF_H
#define CF_H

#include "mlir/Pass/Pass.h"

namespace mlir::cf {

std::unique_ptr<Pass> createCFIndexToI64Pass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace mlir::cf

#endif // CF_H
