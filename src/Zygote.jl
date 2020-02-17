module Zygote

using LinearAlgebra, Statistics
using LinearAlgebra: copytri!, AbstractTriangular

import ZygoteRules: @adjoint, @adjoint!, AContext, adjoint, _pullback, pullback, literal_getproperty

using IRTools
using MacroTools, Requires
using MacroTools: @forward

export Params, gradient, pullback, @code_grad

include("tools/idset.jl")

include("compiler/reverse.jl")
include("compiler/emit.jl")
include("compiler/interface.jl")
include("compiler/show.jl")

include("lib/grad.jl")
include("lib/lib.jl")
include("lib/number.jl")
include("lib/base.jl")
include("lib/array.jl")
include("lib/buffer.jl")
include("lib/broadcast.jl")
include("lib/nnlib.jl")
include("lib/forward.jl")
include("lib/utils.jl")
@init @require Distances="b4f34e82-e78d-54a5-968a-f98e89d6e8f7" include("lib/distances.jl")
@init @require StatsFuns="4c63d2b9-4356-54db-8cca-17b64c39e42c" include("lib/statsfuns.jl")

# we need to define this late, so that the genfuncs see lib.jl
include("compiler/interface2.jl")

include("profiler/Profile.jl")

@init @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
  include("flux.jl")
end

precompile() = include(joinpath(@__DIR__, "precompile.jl"))

# precompile()
@init Requires.isprecompiling() || precompile()

# helps to work around 265-y issues
function refresh()
  include(joinpath(@__DIR__, "compiler/interface2.jl"))
  precompile()
  return
end

macro profile(ex)
  @capture(ex, f_(x__)) || error("@profile f(args...)")
  quote
    _, back = _pullback($(esc(f)), $(esc.(x)...))
    Profile.juno(Profile.profile(back))
  end
end

end # module
