module Zygote

using MacroTools

export forward, @code_grad

include("compiler/ir.jl")
include("compiler/interpret.jl")
include("compiler/reverse.jl")
include("compiler/emit.jl")
include("compiler/reflection.jl")
include("compiler/interface.jl")

include("lib/lib.jl")
include("lib/real.jl")
include("lib/array.jl")

# 1. we need to define this late, so that the genfuncs see lib.jl
# 2. we make it optional until more tests pass in compiled mode
function compiled()
  include(joinpath(@__DIR__, "compiler/interface2.jl"))
end

end # module
