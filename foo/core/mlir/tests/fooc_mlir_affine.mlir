// RUN: fooc %s -emit=mlir-affine | FileCheck %s

func @main() {
  %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
  foo.print %0 : !foo.foo
  foo.return
}
// CHECK-LABEL: func @main() {
// CHECK:   %cst = constant 1.000000e+00 : f64
// CHECK:   %0 = alloc() : memref<f64>
// CHECK:   affine.store %cst, %0[] : memref<f64>
// CHECK:   foo.print %0 : memref<f64>
// CHECK:   dealloc %0 : memref<f64>
// CHECK:   return
// CHECK: }
