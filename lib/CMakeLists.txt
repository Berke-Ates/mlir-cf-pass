add_mlir_dialect_library(CFIndexToInt
  CFIndexToInt.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  MLIRCFIndexToIntPassIncGen
)

target_link_libraries(CFIndexToInt PUBLIC MLIRIR)
