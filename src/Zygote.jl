__precompile__(false)

module Zygote

using IRTools
using MacroTools, Requires
using MacroTools: @forward

export forward, @code_grad

include("tools/idset.jl")
include("tools/ir.jl")
include("tools/slots.jl")
include("tools/reflection.jl")

include("compiler/reverse.jl")
include("compiler/emit.jl")
include("compiler/interface.jl")

include("lib/lib.jl")
include("lib/real.jl")
include("lib/base.jl")
include("lib/array.jl")

include("compiler/interface2.jl")

@init @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("flux.jl")

# we need to define this late, so that the genfuncs see lib.jl
# also helps to work around 265-y issues
refresh() = include(joinpath(@__DIR__, "compiler/interface2.jl"))

@init refresh()

end # module
