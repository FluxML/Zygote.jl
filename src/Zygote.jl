module Zygote

using LinearAlgebra
using LinearAlgebra: copytri!

# This flag enables Zygote to grab extra type inference information during
# compiles. When control flow is present, this can give gradient code a
# performance boost.

# HOWEVER, this is not Jameson-approved, nor well supported by the compiler, and
# has several caveats. Recursion will cause inference to stack overflow.
# Gradient redefinitions may result in ugly type errors. And Jameson *will* know.
const usetyped = get(ENV, "ZYGOTE_TYPED", false) == "true"

using IRTools
using MacroTools, Requires
using MacroTools: @forward

export Params, gradient, derivative, forward, @code_grad

include("tools/idset.jl")
include("tools/ir.jl")
include("tools/reflection.jl")
include("tools/fillarray.jl")

include("compiler/reverse.jl")
include("compiler/emit.jl")
include("compiler/interface.jl")

include("forward/Forward.jl")

include("lib/grad.jl")
include("lib/lib.jl")
include("lib/real.jl")
include("lib/complex.jl")
include("lib/base.jl")
include("lib/array.jl")
include("lib/nnlib.jl")
include("lib/broadcast.jl")

# we need to define this late, so that the genfuncs see lib.jl
include("compiler/interface2.jl")
usetyped || include("precompile.jl")

include("profiler/Profile.jl")

@init @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("flux.jl")

# helps to work around 265-y issues
function refresh()
  include(joinpath(@__DIR__, "compiler/interface2.jl"))
  usetyped || include(joinpath(@__DIR__, "precompile.jl"))
  return
end

macro profile(ex)
  @capture(ex, f_(x__)) || error("@profile f(args...)")
  quote
    _, back = _forward($(esc(f)), $(esc.(x)...))
    Profile.juno(Profile.profile(back))
  end
end

end # module
