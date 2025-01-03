# ToDo: Move some of this to ZygoteRules, or move unthunk_tangent for Tuple and NamedTuple from
# Zygote rules here?
function unthunk_tangent end
@inline unthunk_tangent(x::AbstractThunk) = wrap_chainrules_output(unthunk(x))
@inline unthunk_tangent(x::NTuple{N,<:Number}) where N = x
@inline unthunk_tangent(x::AbstractArray{<:Number,N}) where N = x
@inline unthunk_tangent(x::AbstractArray) = map(unthunk_tangent, x)
unthunk_tangent(d::IdDict) = IdDict([unthunk_tangent(k) => unthunk_tangent(v) for (k, v) in d])
@non_differentiable unthunk_tangent(::IdDict)


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
function has_chain_rrule(T, world)
  config_T, arg_Ts = Iterators.peel(T.parameters)
  configured_rrule_m = meta(Tuple{typeof(rrule), config_T, arg_Ts...}; world)
  is_ambig = configured_rrule_m === nothing  # this means there was an ambiguity error, on configured_rrule


  if !is_ambig && _is_rrule_redispatcher(configured_rrule_m.method)
    # The config is not being used:
    # it is being redispatched without config, so we need the method it redispatches to
    rrule_m = meta(Tuple{typeof(rrule), arg_Ts...}; world)
    # Thus any no_rrule that might apply must also not have a config because if there was a
    # no_rrule with a config that applied then there would also be a rrule with config that applied
    no_rrule_m = meta(Tuple{typeof(ChainRulesCore.no_rrule), arg_Ts...}; world)
  else
    # Not being redispatched: it does have a config
    rrule_m = configured_rrule_m
    # Thus any no_rrule that might apply must also have a config because if it applied
    # it will be identical, and if it doesn't we don't care what it is.
    no_rrule_m = meta(Tuple{typeof(ChainRulesCore.no_rrule), config_T, arg_Ts...}; world)
  end

  is_ambig |= rrule_m === nothing  # this means there was an ambiguity error on unconfigured rrule

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


  if !is_ambig && matching_cr_sig(no_rrule_m, rrule_m)  # Not ambiguous, and opted-out.
    # Return instance for configured_rrule_m as that will be invalidated 
    # directly if configured rule added, or indirectly if unconfigured rule added
    # Do not need an edge for `no_rrule` as no addition of methods to that can cause this
    # decision to need to be revisited (only changes to `rrule`), since we are already not
    # using the rrule, so not using more rules wouldn't change anything.
    return false, configured_rrule_m.instance
  else
    # Either is ambiguous, and we should try to use it, and then error
    # or we are uses a rrule, no need to add any edges for `rrule`, as it will generate 
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
matching_cr_sig(::Any, ::Nothing) = false  # ambiguous https://github.com/FluxML/Zygote.jl/issues/1234

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
@inline wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)
# Zygote convention: even if many AbstractZero partials (i.e. multi-input function), make just 1 nothing.
@inline wrap_chainrules_output(x::Tuple{Vararg{ChainRules.AbstractZero}}) = nothing
@inline wrap_chainrules_output(x::ChainRules.AbstractZero) = nothing
@inline wrap_chainrules_output(x::ChainRulesCore.NotImplemented) = nothing
for T_outer in (:Tuple, :NamedTuple)
  # we create separate methods rather than using a `Union` + an `if` so that we avoid a
  # branch that changes output type, because nested AD on that kinda thing makes Zygote less
  # than happy.
  @eval @inline function wrap_chainrules_output(x::ChainRules.Tangent{P, T}) where {P, T<:$T_outer}
    xp = map(wrap_chainrules_output, canonicalize(x))
    ChainRulesCore.backing(xp)  # this is accessing ChainRulesCore internals, but it is prob safe enough, and it is fastest
  end
end
wrap_chainrules_output(dxs::AbstractArray{<:Number}) = dxs
wrap_chainrules_output(dxs::AbstractArray{<:AbstractArray{<:Number}}) = dxs
wrap_chainrules_output(dxs::AbstractArray) = map(wrap_chainrules_output, dxs)
#=
# As an optimisation, we can convert by `reinterpret` for bitstypes, e.g. arrays of tuples of numbers
@inline function wrap_chainrules_output(dxs::AbstractArray{<:ChainRules.Tangent{<:Any, B}}) where {B}
  if isbitstype(B)
    # B is the backing type. It still contains NoTangent etc, which need converting to Nothing
    reinterpret(wrap_chainrules_output(B), dxs)
  else
    map(wrap_chainrules_output, dxs)
  end
end
wrap_chainrules_output(::Type{<:AbstractZero}) = Nothing
wrap_chainrules_output(::Type{NamedTuple{L,T}}) where {L,T} = NamedTuple{L,wrap_chainrules_output(T)}
@generated function wrap_chainrules_output(::Type{T}) where T<:Tuple
  inner = map(wrap_chainrules_output, T.parameters)
  :(Tuple{$(inner...)})
end
=#

"""
    wrap_chainrules_input(dx)

Convert `dx` from the format Zygote uses internally to differentials types ChainRules uses.
"""
@inline wrap_chainrules_input(dx) = dx
@inline wrap_chainrules_input(::Nothing) = ChainRules.ZeroTangent()
@inline wrap_chainrules_input(::Tuple{Vararg{Nothing}}) = ChainRules.ZeroTangent()
@inline wrap_chainrules_input(::AbstractArray{Nothing}) = ChainRules.ZeroTangent()
@inline function wrap_chainrules_input(dxs::Union{Tuple, NamedTuple})
  xp = map(wrap_chainrules_input, dxs)
  # This produces Tangent{Any} since it does not get to see the primal, `x`.
  ChainRulesCore.Tangent{Any, typeof(xp)}(xp)
end
# For mutable types, including x=Ref(1), Zygote makes Ref{Any}(::NamedTuple)
@inline wrap_chainrules_input(dx::Ref) = wrap_chainrules_input(dx[])
# For arrays, whitelist the safe ones, but always look inside Any[]:
@inline wrap_chainrules_input(dxs::AbstractArray{<:Number}) = dxs
@inline wrap_chainrules_input(dxs::AbstractArray{<:AbstractArray{<:Number}}) = dxs
@inline wrap_chainrules_input(dxs::AbstractArray{<:Union{Nothing,T}}) where T <: Number = map(x -> x === nothing ? zero(T) : x, dxs)
@inline wrap_chainrules_input(dxs::AbstractArray) = map(wrap_chainrules_input, dxs)

#=
# Could `reinterpret` instead here? See issue 1112. 
# One easy case, might be this:
@inline wrap_chainrules_input(xs::Base.ReinterpretArray{<:NamedTuple, <:Tangent}) = parent(xs)

# This is for `z2d` reinterpret below:
wrap_chainrules_input(::Type{Nothing}) = NoTangent
wrap_chainrules_input(::Type{NamedTuple{L,T}}) where {L,T} = NamedTuple{L,wrap_chainrules_input(T)}
@generated function wrap_chainrules_input(::Type{T}) where T<:Tuple
  inner = map(wrap_chainrules_input, T.parameters)
  :(Tuple{$(inner...)})
end
=#

"""
  _project(x, dx)

Uses `ChainRulesCore.ProjectTo` to standardise the gradient `dx` for type & shape.
Also handles some Zygote-specific corrections, such as `x::Array, dx::Tuple`.
Safe to apply to arbitrary input.
"""
@inline function _project(x, dx)
  wrap_chainrules_output(ProjectTo(x)(zygote2differential(dx, x)))
end

# Restore splatted arrays
_project(x::AbstractArray, dx::Tuple) = _project(x, reshape(collect(dx), axes(x)))

# Piracy:
# CRC likes Tangent{AbstractArray}, but Zygote makes Tangent{Any}
# in particular this would hit https://github.com/JuliaDiff/ChainRulesCore.jl/blob/2ec2549b73b22bc08f554dae864fb650cfb9c3d7/src/projection.jl#L139
# if we were not losing track of the Primal in the Tangent
# This type piracy is just giving up that safety check.
(project::ProjectTo{AbstractArray})(dx::Tangent) = dx

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

    ad_pullback(Δ) = zygote2differential(
        pb(wrap_chainrules_output(unthunk_tangent(Δ))),
        f_args)
    return y, ad_pullback
end

"""
    zygote2differential(dx, primal)

Convert input `dx` from the Zygote format to the ChainRules differential types.
This is similar to `wrap_chainrules_input(dx)`, but because it gets `primal::T`,
it can turn `NamedTuple`s into `Tangent{T}(...)` not `Tangent{Any}(...)`.
"""
zygote2differential(x, primal) = z2d(x, primal)
zygote2differential(::Nothing, ::Any) = NoTangent()
zygote2differential(t::Tuple, primal::Tuple) = map(z2d, t, primal)
zygote2differential(t::Tuple, primal) = (@warn "primal should be a tuple, not $primal"; return t)

z2d(::Nothing, ::Any) = NoTangent()
z2d(::Tuple{Vararg{Nothing}}, ::Tuple) = NoTangent()  # collapse all-zero case
z2d(dx, ::Any) = dx
z2d(dx::AbstractArray{<:Number}, primal::AbstractArray) = dx
z2d(dx::AbstractArray{<:AbstractArray{<:Number}}, primal::AbstractArray) = dx
z2d(dx::AbstractArray, primal::AbstractArray) = map(z2d, dx, primal)
#=
# As an optimisation, we can convert by `reinterpret` for bitstypes, e.g. arrays of tuples of numbers
function z2d(dx::AbstractArray{S}, primal::AbstractArray{P}) where {S,P}
  if isbitstype(S)
    T = wrap_chainrules_input(S)
    reinterpret(Tangent{P,T}, dx)
  else
    map(z2d, dx, primal)
  end
end
=#

# Note: this should never be hit if we are converting things right, but it seems to be
# happening in the wild for sufficiently weird functions/types.
# This fixes most (all?) cases, but it would be good to find what we miss.
z2d(x::Union{AbstractZero, Tangent}, ::Any) = return x

function z2d(delta::Tuple, primal::Tuple)
  backing = map(z2d, delta, primal)
  if backing isa Tuple{Vararg{AbstractZero}}
    return NoTangent()  # collapse all-zero case
  else
    return canonicalize(Tangent{typeof(primal), typeof(backing)}(backing))
  end
end

# Dict handling in Zygote is a mess... should this become a  `Tangent{Dict,Dict}` ?
# Right now it uses a NamedTuple but not for fields of the AbstractDict struct
z2d(dx::NamedTuple, primal::AbstractDict) = dx

function _z2d_struct_fallback(delta::NamedTuple, primal::T) where T
  fnames = fieldnames(T)
  deltas = map(n -> get(delta, n, nothing), fnames)
  primals = map(n -> getfield(primal, n), fnames)
  inner = map(z2d, deltas, primals)  # recurse into fields
  if inner isa Tuple{Vararg{AbstractZero}}
    return NoTangent()  # collapse all-zero case
  else
    backing = NamedTuple{fnames}(inner)
    return Tangent{T, typeof(backing)}(backing)
  end
end

function z2d(delta::NamedTuple, primal::T) where T  # arbitrart struct
  if @generated
    fnames = fieldnames(T)
    N = length(fnames)
    deltas = [ :($(Symbol(:delta_, fname)) = get(delta, $(QuoteNode(fname)), nothing)) for fname in fnames ]
    primals = [ :($(Symbol(:primal_, fname)) = getfield(primal, $(QuoteNode(fname)))) for fname in fnames ]
    inner = Expr(:tuple, [ :(z2d($(Symbol(:delta_, fname)), $(Symbol(:primal_, fname)))) for fname in fnames ]...)
    return quote
      $(deltas...)
      $(primals...)
      inner = $inner
      if inner isa Tuple{Vararg{AbstractZero}}
        return NoTangent()  # collapse all-zero case
      else
        backing = NamedTuple{$fnames}(inner)
        return Tangent{T, typeof(backing)}(backing)
      end
    end
  else
    return _z2d_struct_fallback(delta, primal)
  end
end

# Dict case matches signature for ambiguity reasons:
z2d(dx::NamedTuple{L,S}, primal::AbstractDict) where {L,S<:Tuple{Vararg{Union{Number,Nothing}}}} = dx
# On Julia <= 1.6, this fixes easy cases which do not require recursion into fields, e.g.
# @inferred Zygote.z2d((re=1, im=nothing), 3.0+im)
@generated function z2d(delta::NamedTuple{L,S}, primal::T) where {L,S<:Tuple{Vararg{Union{Number,Nothing}}}, T}
  fnames = fieldnames(T)
  deltas = map(fnames) do n
    i = findfirst(isequal(n), L)
    if i == nothing || S.parameters[i] == Nothing
      :(NoTangent())
    else
      :(delta.$n)
    end
  end
  if all(d -> d == :(NoTangent()), deltas)
    return :(NoTangent())  # collapse all-zero case
  else
    return quote
      backing = NamedTuple{$fnames}(($(deltas...),))
      Tangent{$T, typeof(backing)}(backing)
    end
  end
end

z2d(dx::Ref, primal) = z2d(dx[], primal)  # mutable structs
