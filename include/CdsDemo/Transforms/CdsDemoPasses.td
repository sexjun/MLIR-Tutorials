#ifndef CDS
#define CDS
include "mlir/Pass/PassBase.td"

def CdsDemoPass : Pass<"CdsDemoPass"> {
let summary = "My Pass Summary";
  let description = [{
    Here we can now give a much larger description of `MyPass`, including all of
    its various constraints and behavior.
  }];
}

def ConvertReduceMaxToReudeceMin :  Pass<"ConvertReduceMaxToReudeceMin"> {
    let summary = "My Pass Summary22222";
    let description = [{
        Here we can now give a much larger description of `MyPass`, including all of
        its various constraints and behavior.
        =====test
    }];

}

#endif
