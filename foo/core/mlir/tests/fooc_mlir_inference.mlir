// RUN: fooc %s -emit=mlir-inference | FileCheck %s

func @main() {
  %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
  foo.print %0 : !foo.foo
  foo.return
}
// CHECK-LABEL: func @main()
// CHECK:   %0 = "foo.const"() {value = dense<1.000000e+00> : tensor<f64>} : () -> tensor<f64>
// CHECK:   foo.print %0 : tensor<f64>
// CHECK:   foo.return
// CHECK: }
