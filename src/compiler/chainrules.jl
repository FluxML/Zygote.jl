const chainrules_fallback = which(rrule, Tuple{Any})

"""
  has_chain_rrule(T)

For a type-tuple `T` e.g. `Tuple{typeof(f), Int, Float64}`, checks if there is a `rrule` defined for it.
Excluding the generic fallback.
The first return value is a Bool is whether or not the `rrule` exists.
If it does not, then the second argument is a list of edges to attach to the CodeInfo for a generated function,
such that if a suitable rule is defined later, the generated function will recompile.
"""
function has_chain_rrule(T)
  m = meta(Tuple{typeof(rrule),T.parameters...})
  if m.method === chainrules_fallback
    return false, m.code.edges
  else
    return true, nothing
  end
end

"""
    wrap_chainrules_output(x)

Convert `x` from the differentials types ChainRules uses  to the format Zygote uses internally
(including conjugating complex gradients).
"""
wrap_chainrules_output(x) = conj(unthunk(x))  # For now we are just not going to deal with thunks
wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)
wrap_chainrules_output(x::ChainRules.AbstractZero) = nothing
function wrap_chainrules_output(x::ChainRules.Composite{P, T}) where {P, T}
  T_outer = T <: NamedTuple  ? NamedTuple : Tuple  # must be a Tuple or NamedTuple, don't care about exact parameter types
  xp = map(wrap_chainrules_output, x)
  convert(T_outer, xp)
end


"""
    wrap_chainrules_input(x)

Convert `x` from the format  Zygote uses internally (including conjugated complex gradients)
to differentials types ChainRules uses.
"""
wrap_chainrules_input(x) = conj(x)
wrap_chainrules_input(::Nothing) = ChainRules.Zero()
function wrap_chainrules_input(xs::Union{Tuple, NamedTuple})
  xp = map(wrap_chainrules_input, xs)
  ChainRules.Composite{Any, typeof(xp)}(xp)
end

"""
    wrap_chainrules_pullback(f, args...)

Wrap a chainrule's pullback `f`, converting the format of the inputs (`args`),
and the outputs.
"""
function wrap_chainrules_pullback(pb, args...)
  returun wrap_chainrules_output(pb(wrap_chainrules_input(args)...))
end


"""
    chain_rrule(f, args...)

Returns a the (primal) value of `f(args...)` and a pullback, by invoking `ChainRulesCore.rrule(f, args...)`.
The pullback is appropriately wrapped up to follow Zygote conventions.
"""
function chain_rrule(f, args...)
  y, back = rrule(f, args...)

  # Dispatch here handles chainrules considing pullbacks to have multiple input if Tuple.
  # TODO: this could be removed if: https://github.com/JuliaDiff/ChainRulesCore.jl/issues/152
  zpullback(dy) = wrap_chainrules_pullback(back, dy)
  zpullback(dy::Tuple) = wrap_chainrules_pullback(back, dy...)

  # `nothing->nothing` can be deleted after https://github.com/FluxML/Zygote.jl/issues/603
  # though it might be worth keeping as a performance optimization (benchmarking pending)
  zpullback(::Nothing) = nothing

  y, zpullback
end

# Required for nested AD
@adjoint ChainRules.Composite{Any, T}(x::T) where T = ChainRules.Composite{Any, T}(x), x->(x,)
