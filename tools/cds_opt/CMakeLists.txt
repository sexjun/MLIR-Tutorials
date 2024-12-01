message("build cds-opt")
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  CdsPasses # 自己构建的pass
  CdsPatterns # 自己写的patterns
  MLIROptLib # MLIR opt工具的依赖库
  )
#   message("dialect_libs:${dialect_libs}\n\n")

#   message("conversion_libs:${conversion_libs}\n\n")

#   message("extension_libs:${extension_libs}\n\n")

add_mlir_tool(cds-opt cds-opt.cpp DEPENDS ${LIBS})
target_link_libraries(cds-opt
    PRIVATE
    ${LIBS}
)
