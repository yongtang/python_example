// RUN: fooc %s -emit=mlir-llvm | FileCheck %s

func @main() {
  %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
  foo.print %0 : !foo.foo
  foo.return
}
// CHECK-LABEL: llvm.func @free(!llvm<"i8*">)
// CHECK: llvm.mlir.global internal constant @nl("\0A\00")
// CHECK: llvm.mlir.global internal constant @frmt_spec("%f \00")
// CHECK: llvm.func @printf(!llvm<"i8*">, ...) -> !llvm.i32
// CHECK: llvm.func @malloc(!llvm.i64) -> !llvm<"i8*">
// CHECK: llvm.func @main() {
// CHECK:   %0 = llvm.mlir.constant(1.000000e+00 : f64) : !llvm.double
// CHECK:   %1 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %2 = llvm.mlir.null : !llvm<"double*">
// CHECK:   %3 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %4 = llvm.getelementptr %2[%3] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   %5 = llvm.ptrtoint %4 : !llvm<"double*"> to !llvm.i64
// CHECK:   %6 = llvm.mul %1, %5 : !llvm.i64
// CHECK:   %7 = llvm.call @malloc(%6) : (!llvm.i64) -> !llvm<"i8*">
// CHECK:   %8 = llvm.bitcast %7 : !llvm<"i8*"> to !llvm<"double*">
// CHECK:   %9 = llvm.mlir.undef : !llvm<"{ double*, double*, i64 }">
// CHECK:   %10 = llvm.insertvalue %8, %9[0] : !llvm<"{ double*, double*, i64 }">
// CHECK:   %11 = llvm.insertvalue %8, %10[1] : !llvm<"{ double*, double*, i64 }">
// CHECK:   %12 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %13 = llvm.insertvalue %12, %11[2] : !llvm<"{ double*, double*, i64 }">
// CHECK:   %14 = llvm.extractvalue %13[1] : !llvm<"{ double*, double*, i64 }">
// CHECK:   %15 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %16 = llvm.getelementptr %14[%15] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   llvm.store %0, %16 : !llvm<"double*">
// CHECK:   %17 = llvm.mlir.addressof @frmt_spec : !llvm<"[4 x i8]*">
// CHECK:   %18 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %19 = llvm.getelementptr %17[%18, %18] : (!llvm<"[4 x i8]*">, !llvm.i64, !llvm.i64) -> !llvm<"i8*">
// CHECK:   %20 = llvm.mlir.addressof @nl : !llvm<"[2 x i8]*">
// CHECK:   %21 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %22 = llvm.getelementptr %20[%21, %21] : (!llvm<"[2 x i8]*">, !llvm.i64, !llvm.i64) -> !llvm<"i8*">
// CHECK:   %23 = llvm.extractvalue %13[1] : !llvm<"{ double*, double*, i64 }">
// CHECK:   %24 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %25 = llvm.getelementptr %23[%24] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   %26 = llvm.load %25 : !llvm<"double*">
// CHECK:   %27 = llvm.call @printf(%19, %26) : (!llvm<"i8*">, !llvm.double) -> !llvm.i32
// CHECK:   %28 = llvm.extractvalue %13[0] : !llvm<"{ double*, double*, i64 }">
// CHECK:   %29 = llvm.bitcast %28 : !llvm<"double*"> to !llvm<"i8*">
// CHECK:   llvm.call @free(%29) : (!llvm<"i8*">) -> ()
// CHECK:   llvm.return
// CHECK: }
