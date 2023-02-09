ignore_sig(T) = all(T -> T <: Type, T.parameters)

function edge!(m::IRTools.Meta, edge::Core.MethodInstance)
  m.code.edges === nothing && (m.code.edges = Core.MethodInstance[])
  push!(m.code.edges, edge)
  return
end

@generated function _pullback(ctx::AContext, f, args...)
  # Try using ChainRulesCore
  if is_kwfunc(f, args...)
    # if it is_kw then `args[1]` are the keyword args, `args[2]` is actual function
    cr_T = Tuple{ZygoteRuleConfig{ctx}, args[2:end]...}
    chain_rrule_f = :chain_rrule_kw
  else
    cr_T = Tuple{ZygoteRuleConfig{ctx}, f, args...}
    chain_rrule_f = :chain_rrule
  end

  hascr, cr_edge = has_chain_rrule(cr_T)
  hascr && return :($chain_rrule_f(ZygoteRuleConfig(ctx), f, args...))

  # No ChainRule, going to have to work it out.
  T = Tuple{f,args...}
  ignore_sig(T) && return :(f(args...), Pullback{$T}(()))

  g = try
    _generate_pullback_via_decomposition(T)
  catch e
    rethrow(CompileError(T,e))
  end
  g === nothing && return :(f(args...), Pullback{$T}((f,)))
  meta, forw, _ = g
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  forw = varargs!(meta, forw, 3)
  # verify(forw)
  forw = slots!(pis!(inlineable!(forw)))
  # be ready to swap to using chainrule if one is declared
  cr_edge !== nothing && edge!(meta, cr_edge)
  return update!(meta.code, forw)
end

@generated function (j::Pullback{T})(Δ) where T
  ignore_sig(T) && return :nothing
  g = try
    _generate_pullback_via_decomposition(T)
  catch e
    rethrow(CompileError(T,e))
  end
  if g === nothing
    Δ == Nothing && return :nothing
    return :(error("Non-differentiable function $(repr(j.t[1]))"))
  end
  meta, _, back = g
  argnames!(meta, Symbol("#self#"), :Δ)
  # verify(back)
  back = slots!(inlineable!(back))
  return update!(meta.code, back)
end
