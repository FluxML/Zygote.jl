function _lookup_grad(T)
  (meta = typed_meta(T)) == nothing && return
  grad_ir(IRCode(meta), varargs = meta.method.isva)
  forw, back = stacks!(grad_ir(IRCode(meta), varargs = meta.method.isva), T)
  meta, forw, back
end

struct Context
end

struct J{S,T}
  t::T
end

J{S}(x) where S = J{S,typeof(x)}(x)

Base.show(io::IO, j::J{S}) where S = print(io, "J{$(S.parameters[1])}(...)")

@generated function _forward(ctx::Context, f, args...)
  T = Tuple{f,args...}
  (g = _lookup_grad(T)) == nothing && return :(f(args...), J{$T}(()))
  meta, forw, _ = g
  pushfirst!(forw.argtypes, typeof(_forward), Context)
  forw = varargs!(meta, forw, 3)
  update!(meta, forw)
  meta.code.slotnames = [Symbol("#self#"), :ctx, :f, :args]
  return meta.code
end

@generated function (j::J{T})(Δ) where T
  meta, _, back = _lookup_grad(T)
  meta.code.slottypes = Any[j, Δ]
  meta.code.slotnames = Any[Symbol("#self#"), :Δ]
  update!(meta, back)
  # Enable type inference
  meta.code.inferred = false
  meta.code.ssavaluetypes = length(meta.code.ssavaluetypes)
  return meta.code
end

# Interpreted mode

function _forward(ctx::Context, f, args...)
  T = typesof(f, args...)
  (g = _lookup_grad(T)) == nothing &&
    return f(args...), Δ -> error("Undifferentiable function $f")
  meta, forw, back = g
  nargs = meta.method.nargs
  meta.method.isva && (args = (args[1:nargs-2]...,args[nargs-1:end]))
  y, c = interpret(forw, _forward, ctx, f, args..., sparams = [meta.static_params...])
  return y, Δ -> interpret(back, c, Δ)
end

_forward(args...) = _forward(Context(), args...)

function forward(f, args...)
  y, back = _forward(f, args...)
  y, function (Δ)
    Δ = back(Δ)
    Δ == nothing && return
    Base.tail(Δ)
  end
end

function gradient(f, args...)
  y, J = forward(f, args...)
  y isa Real || error("Function output is not scalar")
  return J(1)
end
