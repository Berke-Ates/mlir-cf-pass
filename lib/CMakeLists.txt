add_mlir_dialect_library(CFIndexToI64
  CFIndexToI64.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  MLIRCFIndexToI64PassIncGen
)

target_link_libraries(CFIndexToI64 PUBLIC MLIRIR)
