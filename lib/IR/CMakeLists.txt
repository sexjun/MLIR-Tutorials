message("IR")
add_mlir_library(Toy toy.cpp DEPENDS MLIRToyIncGen MLIRToyTransformsIncGen ToyInterfacesIncGen)
