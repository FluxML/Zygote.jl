module Zygote

using Base.Meta, JSON
using Core: SSAValue, GotoNode

include("ir.jl")
include("reverse.jl")

end # module
