using IRTools: varargs!, inlineable!, pis!, slots!
using IRTools.Inner: argnames!, update!

ignore_sig(T) = all(T -> T <: Type, T.parameters)

@generated function _pullback(ctx::AContext, f, args...)
  T = Tuple{f,args...}
  ignore(T) && return :(f(args...), Pullback{$T}(()))
  hascr, cr_edges = has_chain_rrule(T)
  hascr && return :(chain_rrule(f, args...))
  g = try _lookup_grad(T) catch e e end
  !(g isa Tuple) && return :(f(args...), Pullback{$T}((f,)))
  meta, forw, _ = g
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  forw = varargs!(meta, forw, 3)
  forw = slots!(pis!(inlineable!(forw)))
  append!(meta.code.edges, cr_edges)  # be ready to swap to using chainrule if one is declared
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
