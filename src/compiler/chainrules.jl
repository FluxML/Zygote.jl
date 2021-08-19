struct ZygoteRuleConfig{CTX<:AContext} <: RuleConfig{Union{HasReverseMode,NoForwardsMode}}
  context::CTX
end
ZygoteRuleConfig() = ZygoteRuleConfig(Context())


_is_rrule_redispatcher(m::Method) = m.sig == Tuple{typeof(rrule), RuleConfig, Vararg}

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
  configured_rrule_m = meta(Tuple{typeof(rrule), config_T, arg_Ts...})
  if _is_rrule_redispatcher(configured_rrule_m.method)
    # The config is not being used:
    # it is being redispatched without config, so we need the method it redispatches to
    rrule_m = meta(Tuple{typeof(rrule), arg_Ts...})
    # Thus any no_rrule that might apply must also not have a config because if there was a
    # no_rrule with a config that applied then there would also be a rrule with config that applied
    no_rrule_m = meta(Tuple{typeof(ChainRulesCore.no_rrule), arg_Ts...})
  else
    # Not being redispatched: it does have a config
    rrule_m = configured_rrule_m
    # Thus any no_rrule that might apply must also have a config because if it applied
    # it will be identical, and if it doesn't we don't care what it is.
    no_rrule_m = meta(Tuple{typeof(ChainRulesCore.no_rrule), config_T, arg_Ts...})
  end

  # To understand why we only need to check if the sigs match between no_rrule_m and rrule_m
  # in order to decide if to use, one must consider the following facts:
  # - for every method in `no_rrule` there is a identical one in `rrule` that returns nothing
  # - this includes the general fallback `rrule(::Any...)=nothing`.
  # - a configured rrule/no_rrule is always more specific than a otherwise equivalent unconfigured rrule/no_rrule
  #  
  # Consider the following truth table, for what can occur:
  # rrule: fallback, no_rrule: fallback =>  matches => do not use rrule.
  # rrule: specific, no_rrule: fallback => !matches => do use rrule, as haven't opted out.
  # rrule: fallback, no_rrule: specific =>  IMPOSSIBLE, every no_rule is identical to some rrule
  # rrule: specific, no_rrule: specific =>  matches => do not use rrule as opted out
  # rrule: specific, no_rrule: general  => !matches => do use rrule as a more specific rrule takes preciedent over more general opted out
  # rrule: general , no_rrule: specific =>  IMPOSSIBLE, every no_rule us identical to some rrule so can't have a more general rrule being hit, as the specific one would hit first
  #
  # Note that the fallback cases are the same outcome as the general cases as fallback is just most general.
  # It can be seen that checking if it matches is the correct way to decide if we should use the rrule or not.


  do_not_use_rrule = matching_cr_sig(no_rrule_m, rrule_m)
  if do_not_use_rrule
    # Return instance for configured_rrule_m as that will be invalidated 
    # directly if configured rule added, or indirectly if unconfigured rule added
    # Do not need an edge for `no_rrule` as no addition of methods to that can cause this
    # decision to need to be revisited (only changes to `rrule`), since we are already not
    # using the rrule, so not using more rules wouldn't change anything.
    return false, configured_rrule_m.instance
  else
    # Otherwise found a rrule, no need to add any edges for `rrule`, as it will generate 
    # code with natural edges if a new method is defined there.
    # We also do not need an edge to `no_rrule`, as any time a method is added to `no_rrule`
    # a corresponding method is added to `rrule` (to return `nothing`), thus we will already
    # be revisiting this decision when a new opt-out is added.
    return true, nothing
  end
end

matching_cr_sig(t, s) = matching_cr_sig(t.method.sig, s.method.sig)
matching_cr_sig(::DataType, ::UnionAll) = false
matching_cr_sig(::UnionAll, ::DataType) = false
matching_cr_sig(t::Type, s::Type) = type_tuple_tail(t) == type_tuple_tail(s)
 
type_tuple_tail(d::DataType) = Tuple{d.parameters[2:end]...}
function type_tuple_tail(d::UnionAll)
    body = Base.unwrap_unionall(d)
    body_tt = type_tuple_tail(body)
    return Base.rewrap_unionall(body_tt, d)
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
@inline wrap_chainrules_output(x) = x
@inline wrap_chainrules_output(x::AbstractThunk) = wrap_chainrules_output(unthunk(x))  # For now we are just not going to deal with thunks
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
    ChainRulesCore.backing(xp)  # this is accessing ChainRulesCore internals, but it is prob safe enough, and it is fastest
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
  _project(x)(dx)
  _project(x, dx)

The function `_project(x)` returns a projector, which standardises the gradient `dx` for type & shape.
Uses `ChainRulesCore.ProjectTo`, but is safe to apply to arbitrary input.
The two-argument `_project(x, dx)` applies this immediately.
"""
@inline _project(x) = identity  # fallback: do nothing!
@inline _project(x::Numeric) = wrap_chainrules_output ∘ ProjectTo(x) ∘ wrap_chainrules_input
@inline _project(x::Ref{<:Numeric}) = wrap_chainrules_output ∘ ProjectTo(x) ∘ wrap_chainrules_input

@inline _project(x, dx) = _project(x)(dx)

# PIRACY -- some tests hit a matrix of nothings, which doesn't seem to be handled?
(::ChainRulesCore.ProjectTo)(nothing) = ChainRulesCore.NoTangent()

# julia> Zygote.wrap_chainrules_input(nothing)
# ChainRulesCore.ZeroTangent()
#
# julia> Zygote.wrap_chainrules_input([nothing, nothing])
# 2-element Vector{Nothing}:
#  nothing
#  nothing
# 
# But the original case was an array of Union{Int,Nothing}

# Solve some ambiguity:
(::ProjectTo{ChainRulesCore.NoTangent})(::ChainRulesCore.AbstractZero) = NoTangent()

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

function ChainRulesCore.rrule_via_ad(config::ZygoteRuleConfig, f_args...; kwargs...)
    # first check whether there is an `rrule` which handles this directly
    direcct = rrule(config, f_args...; kwargs...)
    direcct === nothing || return direcct

    # create a closure to work around _pullback not accepting kwargs
    # but avoid creating a closure unnecessarily (pullbacks of closures do not infer)
    y, pb = if !isempty(kwargs)
        kwf() = first(f_args)(Base.tail(f_args)...; kwargs...)
        _y, _pb = _pullback(config.context, kwf)
        _y, Δ -> first(_pb(Δ)).f_args  # `first` should be `only`
    else
        _pullback(config.context, f_args...)
    end

    ad_pullback(Δ) = zygote2differential(pb(wrap_chainrules_output(Δ)), f_args)
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
# Note: this should never be hit if we are converting things right, but it seems to be
# happening in the wild for sufficiently weird functions/types.
# This fixes most (all?) cases, but it would be good to find what we miss.
z2d(x::Union{AbstractZero, Tangent}, ::Any) = return x
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
