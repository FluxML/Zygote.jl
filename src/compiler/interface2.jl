using IRTools: varargs!, inlineable!, pis!, slots!
using IRTools.Inner: argnames!, update!

ignore_sig(T) = all(T -> T <: Type, T.parameters)

const chainrules_fallback = which(rrule, Tuple{Any})

function has_chainrule(T)
  m = meta(Tuple{typeof(rrule),T.parameters...})
  if m.method === chainrules_fallback
    return false, m.code.edges
  else
    return true, nothing
  end
end

@generated function _pullback(ctx::AContext, f, args...)
  T = Tuple{f,args...}
  ignore(T) && return :(f(args...), Pullback{$T}(()))
  hascr, cr_edges = has_chainrule(T)
  hascr && return :(rrule(f, args...))
  g = try _lookup_grad(T) catch e e end
  !(g isa Tuple) && return :(f(args...), Pullback{$T}((f,)))
  meta, forw, _ = g
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  forw = varargs!(meta, forw, 3)
  forw = slots!(pis!(inlineable!(forw)))
  append!(meta.code.edges, cr_edges)
  return update!(meta.code, forw)
end

@generated function (j::Pullback{T})(Δ) where T
  ignore(T) && return :nothing
  g = try _lookup_grad(T)
  catch e
    rethrow(CompileError(T,e))
  end
  if g == nothing
    Δ == Nothing && return :nothing
    return :(error("Non-differentiable function $(repr(j.t[1]))"))
  end
  meta, _, back = g
  argnames!(meta, Symbol("#self#"), :Δ)
  back = slots!(inlineable!(back))
  return update!(meta.code, back)
end




#=="""
  chainrules_blacklist(f, args...,)

This is used to disable the use of ChainRule's definitions
for particular functions/methods.

It is not required if a Zygote rule has already been defined directly.
"""==#
chainrules_blacklist(f, args...) = false

# ChainRules does higher-order functions badly
# see https://github.com/JuliaDiff/ChainRules.jl/issues/122
for f in (map, broadcast, mapreduce, mapfoldl, mapfoldr)
    @eval chainrules_blacklist(::typeof($f), args...) = true
end
chainrules_blacklist(::typeof(sum), f, x::AbstractArray{<:Real}) = true
# Except for sum(abs2, xs), that is fine
chainrules_blacklist(::typeof(sum), ::typeof(abs2), x::AbstractArray{<:Real}) = false


#=="""
  _pullback_via_chainrules(pb)

Converts a ChainRules pullback into a Zygote pullback.
`pb` should be a ChainRules pullback, as returned from the second return value of `rrule`
"""==#
function _pullback_via_chainrules(pb)
  function zygote_pullback(Δs...)
    ∂s = pb(Δs...)
    # TODO: Should not unthunk on the way out of a pullback, but rather on way in since
    # that is when we know it is probably going to be used.
    ∂s_zy = map(ChainRules.unthunk, ∂s)
    @info "Invoking via ChainRules" typeof(pb) typeof(∂s) typeof(∂s_zy)
    return ∂s_zy
  end
end
