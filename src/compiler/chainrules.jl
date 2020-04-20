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
wrap_chainrules(x) = unthunk(x)
wrap_chainrules(x::Tuple) = map(wrap_chainrules, x)

function chain_rrule(f, args...)
  y, By = rrule(f, args...)
  back(::Nothing) = nothing
  back(dy) = wrap_chainrules(By(dy))
  return y, back
end
