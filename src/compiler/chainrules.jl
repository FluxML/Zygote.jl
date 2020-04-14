const chainrules_fallback = which(rrule, Tuple{Any})

function has_chainrule(T)
  m = meta(Tuple{typeof(rrule),T.parameters...})
  if m.method === chainrules_fallback
    return false, m.code.edges
  else
    return true, nothing
  end
end

wrap_chainrules(x::Thunk) = x()
wrap_chainrules(x) = x
wrap_chainrules(x::Tuple) = wrap_chainrules.(x)

function chainrule(f, args...)
  y, back = rrule(f, args...)
  y, dy -> wrap_chainrules(back(dy))
end
