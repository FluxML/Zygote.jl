module Zygote

using MacroTools

export forward, @code_grad

include("tools/ir.jl")
include("tools/reflection.jl")

include("compiler/reverse.jl")
include("compiler/emit.jl")
include("compiler/interface.jl")

include("lib/lib.jl")
include("lib/real.jl")
include("lib/array.jl")

# we need to define this late, so that the genfuncs see lib.jl
include("compiler/interface2.jl")
# helps to work around 265-y issues
refresh() = include(joinpath(@__DIR__, "compiler/interface2.jl"))

end # module
