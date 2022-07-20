// RUN: cf-opt --cf-index-to-int %s | mlir-opt --lower-host-to-llvm | FileCheck %s
// CHECK: llvm
module  {
  func.func @name(){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 0 : i32
    %c2 = arith.constant 0 : index
    %c3 = arith.constant 0 : i1
    cf.br ^bb1(%c0, %c1, %c2, %c3 : index, i32, index, i1)
  ^bb1(%0: index, %1: i32, %2: index, %3: i1): 
    %4 = arith.cmpi slt, %0, %2 : index
    %5 = arith.addi %1, %1 : i32
    %6 = arith.addi %3, %3 : i1
    return
  }
}

