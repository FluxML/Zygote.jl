module Zygote

using MacroTools

export forward, @code_grad

include("compiler/ir.jl")
include("compiler/interpret.jl")
include("compiler/reverse.jl")
include("compiler/emit.jl")

include("interface.jl")

include("lib.jl")

end # module
