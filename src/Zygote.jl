module Zygote

using LinearAlgebra, Statistics
using LinearAlgebra: copytri!, AbstractTriangular

import ZygoteRules
import ZygoteRules: @adjoint, @adjoint!, AContext, adjoint, _pullback, pullback,
  literal_getproperty, literal_getfield, unthunk_tangent

using ChainRulesCore
using ChainRules: ChainRules, AbstractThunk, rrule, unthunk, canonicalize
using IRTools
using MacroTools, Requires
using MacroTools: @forward

import Distributed: pmap, CachingPool, workers
export Params, withgradient, gradient, withjacobian, jacobian, hessian, diaghessian, pullback, pushforward, @code_adjoint
export rrule_via_ad

const Numeric{T<:Number} = Union{T, AbstractArray{<:T}}

include("deprecated.jl")
include("tools/buffer.jl")
include("tools/builtins.jl")

include("forward/Forward.jl")
using .Forward

include("compiler/reverse.jl")
include("compiler/emit.jl")
include("compiler/chainrules.jl")
include("compiler/interface.jl")
include("compiler/show.jl")

include("lib/grad.jl")
include("lib/lib.jl")
include("lib/literal_getproperty.jl")
include("lib/number.jl")
include("lib/base.jl")
include("lib/array.jl")
include("lib/buffer.jl")
include("lib/broadcast.jl")
include("lib/forward.jl")
include("lib/utils.jl")
include("lib/range.jl")
include("lib/logexpfunctions.jl")

# we need to define this late, so that the genfuncs see lib.jl
# Move using statements out of this file to help with sysimage building
using IRTools: varargs!, inlineable!, pis!, slots!
using IRTools.Inner: argnames!, update!
include("compiler/interface2.jl")

include("profiler/Profile.jl")


if !isdefined(Base, :get_extension)
  @init @require Distances="b4f34e82-e78d-54a5-968a-f98e89d6e8f7" include("../ext/ZygoteDistancesExt.jl")
  @init @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("../ext/ZygoteTrackerExt.jl")
  @init @require Colors="5ae59095-9a9b-59fe-a467-6f913c188581" include("../ext/ZygoteColorsExt.jl")
end

using InteractiveUtils
precompile() = Requires.@include("precompile.jl")

# helps to work around 265-y issues
function refresh()
  Requires.@include("compiler/interface2.jl")
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

using PrecompileTools
# This caused freezes on early 1.8 patch versions,
# see https://github.com/SciML/DiffEqFlux.jl/issues/783
@static if VERSION < v"1.8" || VERSION >= v"1.8.5"
  @compile_workload precompile()
end

end # module
