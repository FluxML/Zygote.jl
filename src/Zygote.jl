module Zygote

using LinearAlgebra, Statistics
using LinearAlgebra: copytri!, AbstractTriangular

import ZygoteRules
import ZygoteRules: @adjoint, @adjoint!, AContext, adjoint, _pullback, pullback,
  literal_getproperty, literal_getfield, unthunk_tangent

using ChainRulesCore
using ChainRules: ChainRules, AbstractThunk, rrule, unthunk, canonicalize
using IRTools
using MacroTools
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

using InteractiveUtils

macro profile(ex)
  @capture(ex, f_(x__)) || error("@profile f(args...)")
  quote
    _, back = _pullback($(esc(f)), $(esc.(x)...))
    Profile.juno(Profile.profile(back))
  end
end

using PrecompileTools
@compile_workload begin
    include("precompile.jl")
end

end
