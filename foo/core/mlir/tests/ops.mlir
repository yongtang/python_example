// RUN: foo-opt %s | foo-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = foo.foo %{{.*}} : i32
        %res = foo.foo %0 : i32
        return
    }
}
