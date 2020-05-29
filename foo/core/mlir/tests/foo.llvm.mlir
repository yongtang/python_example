// RUN: fooc %s -emit=mlir-llvm | FileCheck %s

func @foo(%arg0: f64) -> f64 {
  %0 = constant 1.0 : f64
  %1 = addf %arg0, %0 : f64
  return %1 : f64
}
// CHECK-LABEL: llvm.func @foo(%arg0: !llvm.double) -> !llvm.double {
// CHECK-NEXT:   %0 = llvm.mlir.constant(1.000000e+00 : f64) : !llvm.double
// CHECK-NEXT:   %1 = llvm.fadd %arg0, %0 : !llvm.double
// CHECK-NEXT:   llvm.return %1 : !llvm.double
// CHECK-NEXT: }
