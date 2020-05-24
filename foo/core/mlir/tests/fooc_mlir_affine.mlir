// RUN: fooc %s -emit=mlir-affine | FileCheck %s

func @main() {
  %0 = foo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
  foo.print %0 : tensor<2x2xf64>
  foo.return
}
// CHECK-LABEL: func @main()
// CHECK: %cst = constant 1.000000e+00 : f64
// CHECK: %cst_0 = constant 2.000000e+00 : f64
// CHECK: %cst_1 = constant 3.000000e+00 : f64
// CHECK: %cst_2 = constant 4.000000e+00 : f64
// CHECK: %0 = alloc() : memref<2x2xf64>
// CHECK: affine.store %cst, %0[0, 0] : memref<2x2xf64>
// CHECK: affine.store %cst_0, %0[0, 1] : memref<2x2xf64>
// CHECK: affine.store %cst_1, %0[1, 0] : memref<2x2xf64>
// CHECK: affine.store %cst_2, %0[1, 1] : memref<2x2xf64>
// CHECK: foo.print %0 : memref<2x2xf64>
// CHECK: dealloc %0 : memref<2x2xf64>
// CHECK: return
// CHECK: }
