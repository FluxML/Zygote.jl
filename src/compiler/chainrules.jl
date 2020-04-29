const chainrules_fallback = which(rrule, Tuple{Any})

function has_chain_rrule(T)
  m = meta(Tuple{typeof(rrule),T.parameters...})
  if m.method === chainrules_fallback
    return false, m.code.edges
  else
    return true, nothing
  end
end


wrap_chainrules_output(x) = conj(unthunk(x))  # For now we are just not going to deal with thunks
wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)
wrap_chainrules_output(x::ChainRules.AbstractZero) = nothing
function wrap_chainrules_output(x::ChainRules.Composite{P, T}) where {P, T}
  T_outer = T <: NamedTuple  ? NamedTuple : Tuple  # must be a Tuple or NamedTuple, don't care about exact parameter types
  xp = map(wrap_chainrules_output, x)
  convert(T_outer, xp)
end

wrap_chainrules_input(x) = conj(x)
wrap_chainrules_input(::Nothing) = ChainRules.Zero()
function wrap_chainrules_input(xs::Union{Tuple, NamedTuple})
  xp = map(wrap_chainrules_input, xs)
  ChainRules.Composite{Any, typeof(xp)}(xp)
end

wrap_chainrules(f, args...) = wrap_chainrules_output(f(wrap_chainrules_input(args)...))



function chain_rrule(f, args...)
  y, back = rrule(f, args...)

  zpullback(dy) = wrap_chainrules(back, dy)
  zpullback(dy::Tuple) = wrap_chainrules(back, dy...)
  # `nothing->nothing` can be deleted after https://github.com/FluxML/Zygote.jl/issues/603
  # though it might be worth keeping as a performance optimization (benchmarking pending)
  zpullback(::Nothing) = nothing

  y, zpullback
end

# Required for nested AD
@adjoint ChainRules.Composite{Any, T}(x::T) where T = ChainRules.Composite{Any, T}(x), x->(x,)
