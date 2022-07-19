// RUN: cf-opt --cf-index-to-int %s | mlir-opt --lower-host-to-llvm | FileCheck %s
// CHECK: llvm
module  {
  func.func @name(){
    %c0 = arith.constant 0 : index
    cf.br ^bb1(%c0 : index)

  ^bb1(%0: index):
    %c1 = arith.constant 1 : i1
    cf.cond_br %c1, ^bb2, ^bb3

  ^bb2:
    %c2 = arith.constant 2 : index
    cf.br ^bb1(%c2 : index)

  ^bb3:
    return
  }
}
