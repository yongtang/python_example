load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "foo_ops_td_files",
    srcs = [
        "ir/foo_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "foo_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/foo_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/foo_ops.cc.inc",
        ),
        (
            "-gen-dialect-decls",
            "ir/foo_dialect.h.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/foo_dialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/foo_ops.td",
    td_srcs = [
        "ir/foo_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/IR/OpAsmInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

cc_library(
    name = "foo",
    srcs = [
        "ir/foo_dialect.h.inc",
        "ir/foo_ops.cc",
        "ir/foo_ops.cc.inc",
        "ir/foo_ops.h.inc",
    ],
    hdrs = [
        "ir/foo_ops.h",
    ],
    deps = [
        ":foo_inc_gen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "foo_dialect_registration",
    srcs = [
        "ir/dialect_registration.cc",
    ],
    deps = [
        ":foo",
    ],
    alwayslink = 1,
)
