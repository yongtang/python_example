// RUN: fooc %s -emit=jit | FileCheck %s

func @main() {
  %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
  foo.print %0 : !foo.foo
  foo.return
}
// CHECK: 1.000000
