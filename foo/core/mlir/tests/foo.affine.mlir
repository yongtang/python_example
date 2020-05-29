// RUN: fooc %s -emit=mlir-affine | FileCheck %s

func @foo(%arg0: f64) -> f64 {
  %0 = constant 1.0 : f64
  %1 = addf %arg0, %0 : f64
  return %1 : f64
}
// CHECK-LABEL: func @foo(%arg0: f64) -> f64 {
// CHECK-NEXT:   %cst = constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = addf %arg0, %cst : f64
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }
