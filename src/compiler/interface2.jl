ignore_sig(T) = all(T -> T <: Type, T.parameters)

function edge!(m::IRTools.Meta, edge::Core.MethodInstance)
  m.code.edges === nothing && (m.code.edges = Core.MethodInstance[])
  push!(m.code.edges, edge)
  return
end

function _generate_pullback(ctx, world, f, args...)
  # Try using ChainRulesCore
  if is_kwfunc(f, args...)
    # if it is_kw then `args[1]` are the keyword args, `args[2]` is actual function
    cr_T = Tuple{ZygoteRuleConfig{ctx}, args[2:end]...}
    chain_rrule_f = :chain_rrule_kw
  else
    cr_T = Tuple{ZygoteRuleConfig{ctx}, f, args...}
    chain_rrule_f = :chain_rrule
  end

  hascr, cr_edge = has_chain_rrule(cr_T, world)
  hascr && return :($chain_rrule_f(ZygoteRuleConfig(ctx), f, args...))

  # No ChainRule, going to have to work it out.
  T = Tuple{f,args...}
  ignore_sig(T) && return :(f(args...), Pullback{$T}(()))

  g = try
    _generate_pullback_via_decomposition(T, world)
  catch e
    if VERSION < v"1.8"
      # work around Julia bug
      rethrow(CompileError(T,e))
    end
    return :(throw($(CompileError(T,e))))
  end
  g === nothing && return :(f(args...), Pullback{$T}((f,)))
  meta, forw, _ = g
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  forw = varargs!(meta, forw, 3)
  # IRTools.verify(forw)
  forw = slots!(pis!(inlineable!(forw)))
  # be ready to swap to using chainrule if one is declared
  cr_edge != nothing && edge!(meta, cr_edge)
  return update!(meta.code, forw)
end

function _generate_callable_pullback(j::Type{<:Pullback{T}}, world, Δ) where T
  ignore_sig(T) && return :nothing
  g = try
    _generate_pullback_via_decomposition(T, world)
  catch e
    if VERSION < v"1.8"
      # work around Julia bug
      rethrow(CompileError(T,e))
    end
    return :(throw($(CompileError(T,e))))
  end
  if g === nothing
    Δ == Nothing && return :nothing
    return :(error("Non-differentiable function $(repr(j.t[1]))"))
  end
  meta, _, back = g
  argnames!(meta, Symbol("#self#"), :Δ)
  # IRTools.verify(back)
  back = slots!(inlineable!(back))
  return update!(meta.code, back)
end

if VERSION >= v"1.10.0-DEV.873"

# on Julia 1.10, generated functions need to keep track of the world age

function _pullback_generator(world::UInt, source, self, ctx, f, args)
  ret = _generate_pullback(ctx, world, f, args...)
  if ret isa Core.CodeInfo
    if isdefined(Base, :__has_internal_change) && Base.__has_internal_change(v"1.12-alpha", :codeinfonargs)
      ret.nargs = 4
      ret.isva = true
    end
    return ret
  end

  stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :ctx, :f, :args), Core.svec())
  stub(world, source, ret)
end

@eval function _pullback(ctx::AContext, f, args...)
  $(Expr(:meta, :generated, _pullback_generator))
  $(Expr(:meta, :generated_only))
end

function _callable_pullback_generator(world::UInt, source, self, Δ)
  ret = _generate_callable_pullback(self, world, Δ)
  if ret isa Core.CodeInfo
    if isdefined(Base, :__has_internal_change) && Base.__has_internal_change(v"1.12-alpha", :codeinfonargs)
      ret.nargs = 2
      ret.isva = false
    end
    return ret
  end

  stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :Δ), Core.svec())
  stub(world, source, ret)
end

@eval function (j::Pullback)(Δ)
  $(Expr(:meta, :generated, _callable_pullback_generator))
  $(Expr(:meta, :generated_only))
end

else

@generated function _pullback(ctx::AContext, f, args...)
  _generate_pullback(ctx, nothing, f, args...)
end

@generated function (j::Pullback)(Δ)
  _generate_callable_pullback(j, nothing, Δ)
end

end
