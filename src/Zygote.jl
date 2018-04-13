module Zygote

using Base.Meta

include("ir.jl")
include("reverse.jl")
include("cleanup.jl")

macro code_grad(ex)
  :(cleanup.(grad_ir($(code_irm(ex)))))
end

end # module
