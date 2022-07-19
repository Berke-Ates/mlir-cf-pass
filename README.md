# MLIR CF Pass
This project implements an out-of-tree optimization tool with a pass resolving 
https://github.com/llvm/llvm-project/issues/55301.

The pass converts index types in the CF operations to I64 by inserting index casts.  
Currently only `cf.br` is supported.  
The output type can be changed in `lib/CFIndexToInt.cpp:46` (will be added as a command flag)

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-sdfg-opt
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Usage
`cf-opt --cf-index-to-int <file>`
