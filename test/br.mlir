// RUN: cf-opt --cf-index-to-int %s | mlir-opt --lower-host-to-llvm | FileCheck %s
// CHECK: llvm
module  {
  func.func @name(){
    %c0 = arith.constant 0 : index
    cf.br ^bb1(%c0: index)
  ^bb1(%0: index): 
    return
  }
}

