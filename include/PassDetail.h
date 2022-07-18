#ifndef CF_PassDetail_H
#define CF_PassDetail_H

#include "SDFG/Dialect/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cf {

#define GEN_PASS_CLASSES
#include "Passes.h.inc"

} // namespace cf
} // end namespace mlir

#endif // CF_PassDetail_H
