const chainrules_fallback = which(rrule, Tuple{Any})

function has_chain_rrule(T)
  m = meta(Tuple{typeof(rrule),T.parameters...})
  if m.method === chainrules_fallback
    return false, m.code.edges
  else
    return true, nothing
  end
end

# For now we are just not going to deal with thunks
wrap_chainrules_output(x) = conj(unthunk(x))
wrap_chainrules_output(x::Tuple) = map(wrap_chainrules_output, x)
function wrap_chainrules_output(x::ChainRules.Composite{P, T}) where {P, T}
  T_outer = T <: NamedTuple  ? NamedTuple : Tuple  # must be a Tuple or NamedTuple, don't care about exact parameter types  
  # Composite supports map as name preserving, and is fast
  xp = map(wrap_chainrules_output, x)
  convert(T_outer, xp)
end

wrap_chainrules_input(x) = conj(x)
wrap_chainrules_input(x::Tuple) = map(wrap_chainrules_input, x)
wrap_chainrules_input(::Nothing) = ChainRules.Zero()
function wrap_chainrules_input(xs::NamedTuple)
  xs_comp = ChainRules.Composite{Any}(xs...)
  # Composite supports map as name preserving, and is fast
  xs_comp_p = map(wrap_chainrules_input, xs_comp)
end


function chain_rrule(f, args...)
  #@info "Using ChainRule" f, typeof.(args)
  y, back = rrule(f, args...)

  zpullback(dy) = wrap_chainrules_output(back(wrap_chainrules_input(dy)))
  # `nothing->nothing` can be deleted after https://github.com/FluxML/Zygote.jl/issues/603
  # though it might be worth keeping as a performance optimization (benchmarking pending)
  zpullback(::Nothing) = nothing

  y, zpullback
end
