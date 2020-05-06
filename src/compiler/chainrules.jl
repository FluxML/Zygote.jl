const chainrules_fallback = which(rrule, Tuple{Any})

"""
  has_chain_rrule(T)

For a type-tuple `T` e.g. `Tuple{typeof(f), Int, Float64}`, checks if there is a `rrule` defined for it.
Excluding the generic fallback.
The first return value is `true` if the `rrule` exists, `false` otherwise.
If it does not, then the second argument is a list of edges to attach to the CodeInfo for a generated function,
such that if a suitable rule is defined later, the generated function will recompile.
"""
function has_chain_rrule(T)
  m = meta(Tuple{typeof(rrule),T.parameters...})
  if m.method !== chainrules_fallback
    # found a rrule, no need to add any edges
    return true, nothing
  end

  # did not find anything, will have to attach edges so it recompiles if one is added
  @static if VERSION >= v"1.3"
    @assert m.code.edges !== nothing
    return false, m.code.edges
  else
    # pre-julia 1.3 there are no edges
    return false, tuple()
  end
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
# Needs `@pure` because else will not run during type inference.
# This is pure enough, the only generic function it calls is in `Core`
# overloading `Core.kwftype` will no doubt break many other things also
Base.@pure is_kwfunc(k, ::Type{<:NamedTuple}, f, args...) = k===Core.kwftype(f)


"""
    wrap_chainrules_output(x)

Convert `x` from the differentials types ChainRules uses  to the format Zygote uses internally
(including conjugating complex gradients).
"""
@inline wrap_chainrules_output(x) = conj(unthunk(x))  # For now we are just not going to deal with thunks
@inline wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)
@inline wrap_chainrules_output(x::ChainRules.AbstractZero) = nothing
@inline function wrap_chainrules_output(x::ChainRules.Composite{P, T}) where {P, T}
  T_outer = T <: NamedTuple  ? NamedTuple : Tuple  # must be a Tuple or NamedTuple, don't care about exact parameter types
  xp = map(wrap_chainrules_output, x)
  convert(T_outer, xp)
end


"""
    wrap_chainrules_input(x)

Convert `x` from the format  Zygote uses internally (including conjugated complex gradients)
to differentials types ChainRules uses.
"""
@inline wrap_chainrules_input(x) = conj(x)
@inline wrap_chainrules_input(::Nothing) = ChainRules.Zero()
@inline function wrap_chainrules_input(xs::Union{Tuple, NamedTuple})
  xp = map(wrap_chainrules_input, xs)
  ChainRules.Composite{Any, typeof(xp)}(xp)
end

"""
    wrap_chainrules_pullback(f, args...)

Wrap a chainrule's pullback `f`, converting the format of the inputs (`args`),
and the outputs.
"""
@inline function wrap_chainrules_pullback(pb, args...)
  return wrap_chainrules_output(pb(wrap_chainrules_input(args)...))
end

# Note we hand-expess the single arg version of this to remove splatting
# because splatting breaks constant folding
# This can be removed after https://github.com/JuliaDiff/ChainRulesCore.jl/issues/152
@inline function wrap_chainrules_pullback(pb, a)
  return wrap_chainrules_output(pb(wrap_chainrules_input(a)))
end


"""
  ZBack{F}(back) <: Function

Wrapper for a ChainRules pullback `back`, that causes it to follow Zygote conventions.
(A functor here is used rather than a closure to avoid boxing issues);
"""
struct ZBack{F} <: Function
  back::F
end
@inline (s::ZBack)(dy) = wrap_chainrules_pullback(s.back, dy)
# Dispatch here handles chainrules considing pullbacks to have multiple input if Tuple.
# TODO: this could be removed if: https://github.com/JuliaDiff/ChainRulesCore.jl/issues/152
@inline (s::ZBack)(dy::Tuple) = wrap_chainrules_pullback(s.back, dy...)
# `nothing->nothing` can be deleted after https://github.com/FluxML/Zygote.jl/issues/603
# though it might be worth keeping as a performance optimization (benchmarking pending)
@inline (s::ZBack)(::Nothing) = nothing

"""
    chain_rrule(f, args...)

Returns a the (primal) value of `f(args...)` and a pullback, by invoking `ChainRulesCore.rrule(f, args...)`.
The pullback is appropriately wrapped up to follow Zygote conventions.
"""
@inline function chain_rrule(f, args...)
  y, back = rrule(f, args...)
  return y, ZBack(back)
end


"""
  chain_rrule_kw(kwf, kwargs, f, args...)

As per [`chain_rrule`](@ref) but with support for kwargs.
`kwf` should be the kwfunc matching to `f`, and `kwargs` are a `NamedTuple` of keyword arguments.
"""
@inline function chain_rrule_kw(kwf, kwargs, f, args...)
  y, back = rrule(f, args...; kwargs...)
  kw_zpullback(dy) = (nothing, nothing, ZBack(back)(dy)...)  # first two nothings are for kwfunc and kwargs
  return y, kw_zpullback
end

# Required for nested AD
@adjoint ChainRules.Composite{Any, T}(x::T) where T = ChainRules.Composite{Any, T}(x), x->(x,)
