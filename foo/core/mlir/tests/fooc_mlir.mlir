// RUN: fooc %s -emit=mlir | FileCheck %s

func @main() {
  %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
  foo.print %0 : !foo.foo
  foo.return
}
// CHECK-LABEL: func @main()
// CHECK:   %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
// CHECK:   foo.print %0 : !foo.foo
// CHECK:   foo.return
// CHECK: }
