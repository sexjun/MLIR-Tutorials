# add_mlir_dialect_library 是一个专门用于创建包含 MLIR 方言 (Dialect) 的库的宏。它不仅定义了一个 C++ 库，还处理了与 MLIR 方言相关的特定逻辑。
# 自动注册方言：这个宏会处理将方言自动注册到 MLIR 系统中的工作。
# 生成合适的命名空间：确保库和方言在合适的命名空间中。
# 特定于方言的初始化：处理方言的初始化工作，使得在使用该方言时，它可以被正确初始化和使用。

# add_mlir_dialect_library(
#   MyDialect
#   SOURCES
#     MyDialect.cpp
#     MyOps.cpp
#   LINK_LIBS
#     MLIRIR
# )



# add_mlir_library 是一个更通用的宏，用于创建 MLIR 项目中的库。它可以用于创建任何类型的库，而不仅仅是包含方言的库。
# 通用库创建：这个宏用于创建通用的 MLIR 库，可以是任何类型的库，而不仅仅是包含方言的库。
# 处理依赖关系：处理库之间的依赖关系，确保它们被正确链接。
# add_mlir_library(
#   MyLibrary
#   SOURCES
#     MyLibrary.cpp
#   LINK_LIBS
#     SomeOtherLibrary
# )


add_mlir_library(Toy toy.cpp DEPENDS MLIRToyIncGen)
add_subdirectory(Transforms)
