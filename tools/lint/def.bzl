load("@bazel_skylib//lib:shell.bzl", "shell")

def _runner_impl(ctx):
    bash_file = ctx.actions.declare_file(ctx.label.name + ".bash")
    substitutions = {
        "@@RUN_PATH@@": shell.quote(ctx.executable._binary.short_path),
        "@@RUN_ARGS@@": shell.quote(ctx.attr.run_args),
    }
    ctx.actions.expand_template(
        template = ctx.file._runner,
        output = bash_file,
        substitutions = substitutions,
        is_executable = True,
    )
    runfiles = ctx.runfiles(files = [ctx.executable._binary])
    return [DefaultInfo(
        files = depset([bash_file]),
        runfiles = runfiles,
        executable = bash_file,
    )]

_clang = rule(
    implementation = _runner_impl,
    attrs = {
        "_binary": attr.label(
            default = "//tools/lint:clang_binary",
            cfg = "host",
            executable = True,
        ),
        "_runner": attr.label(
            default = "run.tpl",
            allow_single_file = True,
        ),
        "run_args": attr.string(default = "--style=google -i"),
    },
    executable = True,
)

_black = rule(
    implementation = _runner_impl,
    attrs = {
        "_binary": attr.label(
            default = "//tools/lint:black_binary",
            cfg = "host",
            executable = True,
        ),
        "_runner": attr.label(
            default = "run.tpl",
            allow_single_file = True,
        ),
        "run_args": attr.string(default = ""),
    },
    executable = True,
)

_pyupgrade = rule(
    implementation = _runner_impl,
    attrs = {
        "_binary": attr.label(
            default = "//tools/lint:pyupgrade_binary",
            cfg = "host",
            executable = True,
        ),
        "_runner": attr.label(
            default = "run.tpl",
            allow_single_file = True,
        ),
        "run_args": attr.string(default = "--py3-only"),
    },
    executable = True,
)

def clang(**kwargs):
    _clang(**kwargs)

def black(**kwargs):
    _black(**kwargs)

def pyupgrade(**kwargs):
    _pyupgrade(**kwargs)
