mlir-opt 包含了MLIR项目中各种优化和转换，通过不同Pass选项可以运行不同的Pass。
mlir-cpu-runner 是官方的JIT即时编译引擎，可以用来对优化和转换的测试，确保每个Pass都正确，写脚本也可以很方便找到那个错误的Pass，具体使用方式可以参考测试文件。
mlir-translate 用于在 MLIR 和其他表示之间进行转换。
