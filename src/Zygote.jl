module Zygote

using Base.Meta

include("ir.jl")
include("reverse.jl")
include("interpret.jl")

macro code_grad(ex)
  :(grad_ir($(code_irm(ex))))
end
include("emit.jl")

end # module
