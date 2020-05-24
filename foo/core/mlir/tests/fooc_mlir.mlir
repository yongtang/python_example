// RUN: fooc %s -emit=mlir | FileCheck %s

func @main() {
  %0 = foo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
  foo.print %0 : tensor<2x2xf64>
  foo.return
}
// CHECK-LABEL: func @main()
// CHECK: %0 = foo.constant dense<{{.*}}> : tensor<2x2xf64>
// CHECK: foo.print %0 : tensor<2x2xf64>
// CHECK: foo.return
// CHECK: }
