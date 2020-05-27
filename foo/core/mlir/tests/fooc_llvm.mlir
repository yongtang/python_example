// RUN: fooc %s -emit=llvm | FileCheck %s

func @main() {
  %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
  foo.print %0 : !foo.foo
  foo.return
}
// CHECK-LABEL: ; ModuleID = 'LLVMDialectModule'
// CHECK: @nl = internal constant [2 x i8] c"\0A\00"
// CHECK: @frmt_spec = internal constant [4 x i8] c"%f \00"
// CHECK: declare i8* @malloc(i64)
// CHECK: declare void @free(i8*)
// CHECK: declare i32 @printf(i8*, ...)
// CHECK: define void @main() !dbg !3 {
// CHECK:   %1 = call i8* @malloc(i64 ptrtoint (double* getelementptr (double, double* null, i64 1) to i64)), !dbg !7
// CHECK:   %2 = bitcast i8* %1 to double*, !dbg !7
// CHECK:   %3 = insertvalue { double*, double*, i64 } undef, double* %2, 0, !dbg !7
// CHECK:   %4 = insertvalue { double*, double*, i64 } %3, double* %2, 1, !dbg !7
// CHECK:   %5 = insertvalue { double*, double*, i64 } %4, i64 0, 2, !dbg !7
// CHECK:   %6 = extractvalue { double*, double*, i64 } %5, 1, !dbg !7
// CHECK:   %7 = getelementptr double, double* %6, i64 0, !dbg !7
// CHECK:   store double 1.000000e+00, double* %7, align 8, !dbg !7
// CHECK:   %8 = extractvalue { double*, double*, i64 } %5, 1, !dbg !9
// CHECK:   %9 = getelementptr double, double* %8, i64 0, !dbg !9
// CHECK:   %10 = load double, double* %9, align 8, !dbg !9
// CHECK:   %11 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %10), !dbg !9
// CHECK:   %12 = extractvalue { double*, double*, i64 } %5, 0, !dbg !7
// CHECK:   %13 = bitcast double* %12 to i8*, !dbg !7
// CHECK:   call void @free(i8* %13), !dbg !7
// CHECK:   ret void, !dbg !10
// CHECK: }
