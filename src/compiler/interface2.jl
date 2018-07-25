@generated function _forward(ctx::Context, f, args...)
  T = Tuple{f,args...}
  ignore(T) && return :(f(args...), J{$T}(()))
  g = try _lookup_grad(T) catch e e end
  !(g isa Tuple) && return :(f(args...), J{$T}((f,)))
  meta, forw, _ = g
  forw = varargs!(meta, forw, 3)
  forw = inlineable!(forw)
  update!(meta, forw)
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  return meta.code
end

@generated function (j::J{T})(Δ) where T
  ignore(T) && return :nothing
  g = _lookup_grad(T)
  if g == nothing
    Δ == Nothing && return :nothing
    return :(error("Non-differentiable function $(j.t[1])"))
  end
  meta, _, back = g
  resize!(back.argtypes, 2)
  argnames!(meta, Symbol("#self#"), :Δ)
  back = inlineable!(back)
  update!(meta, back)
  # Enable type inference
  meta.code.inferred = false
  meta.code.ssavaluetypes = length(meta.code.ssavaluetypes)
  return meta.code
end
