// RUN: fooc %s -emit=llvm | FileCheck %s

func @main() {
  %0 = foo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
  foo.print %0 : tensor<2x2xf64>
  foo.return
}
// CHECK-LABEL: define void @main() !dbg !3 {
// CHECK:   %1 = call i8* @malloc(i64 mul (i64 ptrtoint (double* getelementptr (double, double* null, i64 1) to i64), i64 4)), !dbg !7
// CHECK:   %2 = bitcast i8* %1 to double*, !dbg !7
// CHECK:   %3 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } undef, double* %2, 0, !dbg !7
// CHECK:   %4 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %3, double* %2, 1, !dbg !7
// CHECK:   %5 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %4, i64 0, 2, !dbg !7
// CHECK:   %6 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %5, i64 2, 3, 0, !dbg !7
// CHECK:   %7 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %6, i64 2, 4, 0, !dbg !7
// CHECK:   %8 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %7, i64 2, 3, 1, !dbg !7
// CHECK:   %9 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %8, i64 1, 4, 1, !dbg !7
// CHECK:   %10 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
// CHECK:   %11 = getelementptr double, double* %10, i64 0, !dbg !7
// CHECK:   store double 1.000000e+00, double* %11, align 8, !dbg !7
// CHECK:   %12 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
// CHECK:   %13 = getelementptr double, double* %12, i64 1, !dbg !7
// CHECK:   store double 2.000000e+00, double* %13, align 8, !dbg !7
// CHECK:   %14 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
// CHECK:   %15 = getelementptr double, double* %14, i64 2, !dbg !7
// CHECK:   store double 3.000000e+00, double* %15, align 8, !dbg !7
// CHECK:   %16 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
// CHECK:   %17 = getelementptr double, double* %16, i64 3, !dbg !7
// CHECK:   store double 4.000000e+00, double* %17, align 8, !dbg !7
// CHECK:   br label %18, !dbg !9
//
// CHECK: 18:                                               ; preds = %35, %0
// CHECK:   %19 = phi i64 [ 0, %0 ], [ %37, %35 ]
// CHECK:   %20 = icmp slt i64 %19, 2, !dbg !9
// CHECK:   br i1 %20, label %21, label %38, !dbg !9
//
// CHECK: 21:                                               ; preds = %18
// CHECK:   br label %22, !dbg !9
//
// CHECK: 22:                                               ; preds = %25, %21
// CHECK:   %23 = phi i64 [ 0, %21 ], [ %34, %25 ]
// CHECK:   %24 = icmp slt i64 %23, 2, !dbg !9
// CHECK:   br i1 %24, label %25, label %35, !dbg !9
//
// CHECK: 25:                                               ; preds = %22
// CHECK:   %26 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !9
// CHECK:   %27 = mul i64 %19, 2, !dbg !9
// CHECK:   %28 = add i64 0, %27, !dbg !9
// CHECK:   %29 = mul i64 %23, 1, !dbg !9
// CHECK:   %30 = add i64 %28, %29, !dbg !9
// CHECK:   %31 = getelementptr double, double* %26, i64 %30, !dbg !9
// CHECK:   %32 = load double, double* %31, align 8, !dbg !9
// CHECK:   %33 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %32), !dbg !9
// CHECK:   %34 = add i64 %23, 1, !dbg !9
// CHECK:   br label %22, !dbg !9
//
// CHECK: 35:                                               ; preds = %22
// CHECK:   %36 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @nl, i64 0, i64 0)), !dbg !9
// CHECK:   %37 = add i64 %19, 1, !dbg !9
// CHECK:   br label %18, !dbg !9
//
// CHECK: 38:                                               ; preds = %18
// CHECK:   %39 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 0, !dbg !7
// CHECK:   %40 = bitcast double* %39 to i8*, !dbg !7
// CHECK:   call void @free(i8* %40), !dbg !7
// CHECK:   ret void, !dbg !10
// CHECK: }
