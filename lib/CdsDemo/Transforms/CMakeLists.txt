# pass的配置
add_mlir_library(
    CdsPasses
    CdsDemoPasses.cpp
    DEPENDS
    MLIRCdsDemoPassesIncGen
    CdsPatterns
)
# target_include_directories(CdsPass ${PROJECT_SOURCE_DIR}/include/cds_demo)
