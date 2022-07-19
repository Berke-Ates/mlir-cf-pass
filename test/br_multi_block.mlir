// RUN: cf-opt --cf-index-to-int %s | mlir-opt --lower-host-to-llvm | FileCheck %s
// CHECK: llvm
module  {
  func.func @name(){
    %c0 = arith.constant 0 : index
    cf.br ^bb1(%c0: index)

  ^bb1(%0: index):
    %c1 = arith.constant 1 : index
    cf.br ^bb2(%c1: index)

  ^bb2(%1: index):
    %c2 = arith.constant 2 : index
    cf.br ^bb3(%c2: index)

  ^bb3(%2: index): 
    return
  }
}
