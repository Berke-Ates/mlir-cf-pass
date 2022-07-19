get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib CFIndexToI64)

add_llvm_executable(cf-opt cf-opt.cpp)
llvm_update_compile_flags(cf-opt)

target_link_libraries(cf-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(cf-opt)
