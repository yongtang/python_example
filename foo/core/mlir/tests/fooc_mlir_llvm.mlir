// RUN: fooc %s -emit=mlir-llvm | FileCheck %s

func @main() {
  %0 = foo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
  foo.print %0 : tensor<2x2xf64>
  foo.return
}
// CHECK: llvm.func @free(!llvm<"i8*">)
// CHECK: llvm.mlir.global internal constant @nl("\0A\00")
// CHECK: llvm.mlir.global internal constant @frmt_spec("%f \00")
// CHECK: llvm.func @printf(!llvm<"i8*">, ...) -> !llvm.i32
// CHECK: llvm.func @malloc(!llvm.i64) -> !llvm<"i8*">
// CHECK: llvm.func @main() {
// CHECK:   %0 = llvm.mlir.constant(1.000000e+00 : f64) : !llvm.double
// CHECK:   %1 = llvm.mlir.constant(2.000000e+00 : f64) : !llvm.double
// CHECK:   %2 = llvm.mlir.constant(3.000000e+00 : f64) : !llvm.double
// CHECK:   %3 = llvm.mlir.constant(4.000000e+00 : f64) : !llvm.double
// CHECK:   %4 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %5 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %6 = llvm.mul %4, %5 : !llvm.i64
// CHECK:   %7 = llvm.mlir.null : !llvm<"double*">
// CHECK:   %8 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %9 = llvm.getelementptr %7[%8] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   %10 = llvm.ptrtoint %9 : !llvm<"double*"> to !llvm.i64
// CHECK:   %11 = llvm.mul %6, %10 : !llvm.i64
// CHECK:   %12 = llvm.call @malloc(%11) : (!llvm.i64) -> !llvm<"i8*">
// CHECK:   %13 = llvm.bitcast %12 : !llvm<"i8*"> to !llvm<"double*">
// CHECK:   %14 = llvm.mlir.undef : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %15 = llvm.insertvalue %13, %14[0] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %16 = llvm.insertvalue %13, %15[1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %17 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %18 = llvm.insertvalue %17, %16[2] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %19 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %20 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %21 = llvm.insertvalue %4, %18[3, 0] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %22 = llvm.insertvalue %20, %21[4, 0] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %23 = llvm.insertvalue %5, %22[3, 1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %24 = llvm.insertvalue %19, %23[4, 1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %25 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %26 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %27 = llvm.extractvalue %24[1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %28 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %29 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %30 = llvm.mul %25, %29 : !llvm.i64
// CHECK:   %31 = llvm.add %28, %30 : !llvm.i64
// CHECK:   %32 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %33 = llvm.mul %26, %32 : !llvm.i64
// CHECK:   %34 = llvm.add %31, %33 : !llvm.i64
// CHECK:   %35 = llvm.getelementptr %27[%34] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   llvm.store %0, %35 : !llvm<"double*">
// CHECK:   %36 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %37 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %38 = llvm.extractvalue %24[1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %39 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %40 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %41 = llvm.mul %36, %40 : !llvm.i64
// CHECK:   %42 = llvm.add %39, %41 : !llvm.i64
// CHECK:   %43 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %44 = llvm.mul %37, %43 : !llvm.i64
// CHECK:   %45 = llvm.add %42, %44 : !llvm.i64
// CHECK:   %46 = llvm.getelementptr %38[%45] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   llvm.store %1, %46 : !llvm<"double*">
// CHECK:   %47 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %48 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %49 = llvm.extractvalue %24[1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %50 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %51 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %52 = llvm.mul %47, %51 : !llvm.i64
// CHECK:   %53 = llvm.add %50, %52 : !llvm.i64
// CHECK:   %54 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %55 = llvm.mul %48, %54 : !llvm.i64
// CHECK:   %56 = llvm.add %53, %55 : !llvm.i64
// CHECK:   %57 = llvm.getelementptr %49[%56] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   llvm.store %2, %57 : !llvm<"double*">
// CHECK:   %58 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %59 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %60 = llvm.extractvalue %24[1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %61 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %62 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %63 = llvm.mul %58, %62 : !llvm.i64
// CHECK:   %64 = llvm.add %61, %63 : !llvm.i64
// CHECK:   %65 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %66 = llvm.mul %59, %65 : !llvm.i64
// CHECK:   %67 = llvm.add %64, %66 : !llvm.i64
// CHECK:   %68 = llvm.getelementptr %60[%67] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   llvm.store %3, %68 : !llvm<"double*">
// CHECK:   %69 = llvm.mlir.addressof @frmt_spec : !llvm<"[4 x i8]*">
// CHECK:   %70 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %71 = llvm.getelementptr %69[%70, %70] : (!llvm<"[4 x i8]*">, !llvm.i64, !llvm.i64) -> !llvm<"i8*">
// CHECK:   %72 = llvm.mlir.addressof @nl : !llvm<"[2 x i8]*">
// CHECK:   %73 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %74 = llvm.getelementptr %72[%73, %73] : (!llvm<"[2 x i8]*">, !llvm.i64, !llvm.i64) -> !llvm<"i8*">
// CHECK:   %75 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %76 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %77 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   llvm.br ^bb1(%75 : !llvm.i64)
// CHECK: ^bb1(%78: !llvm.i64):  // 2 preds: ^bb0, ^bb5
// CHECK:   %79 = llvm.icmp "slt" %78, %76 : !llvm.i64
// CHECK:   llvm.cond_br %79, ^bb2, ^bb6
// CHECK: ^bb2:  // pred: ^bb1
// CHECK:   %80 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %81 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %82 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   llvm.br ^bb3(%80 : !llvm.i64)
// CHECK: ^bb3(%83: !llvm.i64):  // 2 preds: ^bb2, ^bb4
// CHECK:   %84 = llvm.icmp "slt" %83, %81 : !llvm.i64
// CHECK:   llvm.cond_br %84, ^bb4, ^bb5
// CHECK: ^bb4:  // pred: ^bb3
// CHECK:   %85 = llvm.extractvalue %24[1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %86 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %87 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:   %88 = llvm.mul %78, %87 : !llvm.i64
// CHECK:   %89 = llvm.add %86, %88 : !llvm.i64
// CHECK:   %90 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %91 = llvm.mul %83, %90 : !llvm.i64
// CHECK:   %92 = llvm.add %89, %91 : !llvm.i64
// CHECK:   %93 = llvm.getelementptr %85[%92] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
// CHECK:   %94 = llvm.load %93 : !llvm<"double*">
// CHECK:   %95 = llvm.call @printf(%71, %94) : (!llvm<"i8*">, !llvm.double) -> !llvm.i32
// CHECK:   %96 = llvm.add %83, %82 : !llvm.i64
// CHECK:   llvm.br ^bb3(%96 : !llvm.i64)
// CHECK: ^bb5:  // pred: ^bb3
// CHECK:   %97 = llvm.call @printf(%74) : (!llvm<"i8*">) -> !llvm.i32
// CHECK:   %98 = llvm.add %78, %77 : !llvm.i64
// CHECK:   llvm.br ^bb1(%98 : !llvm.i64)
// CHECK: ^bb6:  // pred: ^bb1
// CHECK:   %99 = llvm.extractvalue %24[0] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
// CHECK:   %100 = llvm.bitcast %99 : !llvm<"double*"> to !llvm<"i8*">
// CHECK:   llvm.call @free(%100) : (!llvm<"i8*">) -> ()
// CHECK:   llvm.return
// CHECK: }
