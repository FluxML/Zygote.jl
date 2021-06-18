struct ZygoteRuleConfig{CTX<:AContext} <: RuleConfig{Union{HasReverseMode,NoForwardsMode}}
  context::CTX
end
ZygoteRuleConfig() = ZygoteRuleConfig(Context())


const rrule_fallback_method = Base.which(rrule, Tuple{Any, Vararg{Any}})
const rrule_redispatcher_method = Base.which(rrule, Tuple{RuleConfig, Any, Vararg{Any}})

"""
  has_chain_rrule(T)

For a type-tuple `T` e.g. `Tuple{typeof(f), Int, Float64}`, checks if there is a `rrule` defined for it.
Excluding the generic fallback.
The first return value is `true` if the `rrule` exists, `false` otherwise.
If it does not, then the second argument is a list of edges to attach to the CodeInfo for a generated function,
such that if a suitable rule is defined later, the generated function will recompile.
"""
function has_chain_rrule(T)
  config_T, arg_Ts = Iterators.peel(T.parameters)
  m_with_config = meta(Tuple{typeof(rrule), config_T, arg_Ts...})
  if m_with_config.method === rrule_redispatcher_method
    # it is being redispatched without config, so check it that hits the fallback
    m_without_config = meta(Tuple{typeof(rrule), arg_Ts...})
    if m_without_config.method === rrule_fallback_method
      # no rrule exists, return instance for m_with_config as that will be invalidated 
      # directly if configured rule added, or indirectly if unconfigured rule added
      return false, m_with_config.instance
    end
  end
  # otherwise found a rrule, no need to add any edges, as it will generate code with
  # natural edges.
  return true, nothing
end

"""
    is_kwfunc(sigt...)

Determines if `sigt` is the type signature of a kwfunction.
Each element of `sigt` should be a type.
Either the first 3 types are a kwfunc type, a NamedTuple and the matching base function type,
or the first argument is the base function type and it is not a kwfunction.
the remaining types in `sigt` are the types of the argument.

"""
is_kwfunc(::Vararg{Any}) = false
is_kwfunc(k, ::Type{<:NamedTuple}, f, args...) = k===Core.kwftype(f)


"""
    wrap_chainrules_output(x)

Convert `x` from the differentials types ChainRules uses to the format Zygote uses internally.
"""
@inline wrap_chainrules_output(x) = unthunk(x)  # For now we are just not going to deal with thunks
@inline wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)
# Zygote convention: even if many AbstractZero partials (i.e. multi-input function), make just 1 nothing.
@inline wrap_chainrules_output(x::Tuple{Vararg{ChainRules.AbstractZero}}) = nothing
@inline wrap_chainrules_output(x::ChainRules.AbstractZero) = nothing
for T_outer in (:Tuple, :NamedTuple)
  # we create separate methods rather than using a `Union` + an `if` so that we avoid a
  # branch that changes output type, because nested AD on that kinda thing makes Zygote less
  # than happy.
  @eval @inline function wrap_chainrules_output(x::ChainRules.Tangent{P, T}) where {P, T<:$T_outer}
    xp = map(wrap_chainrules_output, canonicalize(x))
    convert($T_outer, xp)
  end
end

"""
    wrap_chainrules_input(x)

Convert `x` from the format Zygote uses internally to differentials types ChainRules uses.
"""
@inline wrap_chainrules_input(x) = x
@inline wrap_chainrules_input(::Nothing) = ChainRules.ZeroTangent()
@inline function wrap_chainrules_input(xs::Union{Tuple, NamedTuple})
  xp = map(wrap_chainrules_input, xs)
  ChainRules.Tangent{Any, typeof(xp)}(xp)
end

"""
  ZBack{F}(back) <: Function

Wrapper for a ChainRules pullback `back`, that causes it to follow Zygote conventions.
(A functor here is used rather than a closure to avoid boxing issues);
"""
struct ZBack{F} <: Function
  back::F
end
@inline (s::ZBack)(dy) = wrap_chainrules_output(s.back(wrap_chainrules_input(dy)))
# `nothing->nothing` can be deleted after https://github.com/FluxML/Zygote.jl/issues/603
# though it might be worth keeping as a performance optimization (benchmarking pending)
@inline (s::ZBack)(::Nothing) = nothing

"""
    chain_rrule(config, f, args...)

Returns a the (primal) value of `f(args...)` and a pullback, by invoking `ChainRulesCore.rrule(f, args...)`.
The pullback is appropriately wrapped up to follow Zygote conventions.
"""
@inline function chain_rrule(config, f, args...)
  y, back = rrule(config, f, args...)
  return y, ZBack(back)
end


"""
  chain_rrule_kw(config, kwf, kwargs, f, args...)

As per [`chain_rrule`](@ref) but with support for kwargs.
`kwf` should be the kwfunc matching to `f`, and `kwargs` are a `NamedTuple` of keyword arguments.
"""
@inline function chain_rrule_kw(config, kwf, kwargs, f, args...)
  y, back = rrule(config, f, args...; kwargs...)
  function kw_zpullback(dy)
    dxs = ZBack(back)(dy)
    if dxs === nothing  # if dxs is nothing, then all partiaols are nothing
      # Zygote convention is a single nothing no mather how partials, if all are nothing
      return nothing
    else
      return (nothing, nothing, dxs...)  # first two nothings are for kwfunc and kwargs
    end
  end
  return y, kw_zpullback
end


function ChainRulesCore.rrule_via_ad(config::ZygoteRuleConfig, f, args...)
    y, pb = _pullback(config.context, f, args...)
    ad_pullback(Δ) = zygote2differential(pb(wrap_chainrules_output(Δ)), (f, args...))
    return y, ad_pullback
end

"""
    zygote2differential(x)

Convert input `x` from the Zygote format to the ChainRules differential types.
"""
zygote2differential(x, primal) = z2d(x, primal)
zygote2differential(::Nothing, ::Any) = NoTangent()
zygote2differential(t::Tuple, primal::Tuple) = map(z2d, t, primal)
zygote2differential(t::Tuple, primal) = (@warn "primal should be a tuple, not $primal"; return t)
z2d(x, ::Any) = x
z2d(::Nothing, ::Any) = NoTangent()
z2d(a::AbstractArray{<:Number}, primal::AbstractArray{T}) where T = a
z2d(a::AbstractArray, primal::AbstractArray{T}) where T = z2d.(a, primal)
z2d(x::Union{AbstractZero, Tangent}, ::Any) = (difftype_warn(x); return x)
function z2d(t::Tuple, primal::Tuple)
  tp::Tuple = map(z2d, t, primal)
  primal_type = typeof(primal)
  return canonicalize(Tangent{primal_type, typeof(tp)}(tp))
end

function z2d(t::NamedTuple, primal)
  primal_type = typeof(primal)
  fnames = fieldnames(primal_type)
  complete_t = NamedTuple{fnames}(fn in keys(t) ? t[fn] : nothing for fn in fnames)
  primals = NamedTuple{fnames}(getfield(primal, fn) for fn in fnames)
  tp::NamedTuple = map(z2d, complete_t, primals)
  return canonicalize(Tangent{primal_type, typeof(tp)}(tp))
end
