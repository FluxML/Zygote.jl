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

end # module
