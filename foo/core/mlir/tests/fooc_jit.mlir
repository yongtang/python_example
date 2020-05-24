// RUN: fooc %s -emit=jit | FileCheck %s

func @main() {
  %0 = foo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
  foo.print %0 : tensor<2x2xf64>
  foo.return
}
// CHECK: 1.000000 2.000000 
// CHECK: 3.000000 4.000000 
