ignore(T) = all(T -> T <: Type, T.parameters)

@generated function _forward(ctx::Context, f, args...)
  T = Tuple{f,args...}
  ignore(T) && return :(f(args...), Pullback{$T}(()))
  g = try _lookup_grad(T) catch e e end
  !(g isa Tuple) && return :(f(args...), Pullback{$T}((f,)))
  meta, forw, _ = g
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  meta.code.slotflags = UInt8[0x0, 0x0, 0x0, 0x0]
  forw = varargs!(meta, forw, 3)
  forw = slots!(pis!(inlineable!(forw)))
  return IRTools.update!(meta, forw)
end

@generated function (j::Pullback{T})(Δ) where T
  ignore(T) && return :nothing
  g = try _lookup_grad(T)
  catch e
    rethrow(CompileError(T,e))
  end
  if g == nothing
    Δ == Nothing && return :nothing
    return :(error("Non-differentiable function $(j.t[1])"))
  end
  meta, _, back = g
  resize!(back.argtypes, 2)
  argnames!(meta, Symbol("#self#"), :Δ)
  meta.code.slotflags = UInt8[0x0, 0x0]
  back = slots!(inlineable!(back))
  return IRTools.update!(meta, back)
end
