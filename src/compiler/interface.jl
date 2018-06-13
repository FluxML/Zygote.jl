function _lookup_grad(T)
  (meta = typed_meta(T)) == nothing && return
  grad_ir(IRCode(meta), varargs = meta.method.isva)
  forw, back = stacks!(grad_ir(IRCode(meta), varargs = meta.method.isva))
  meta, forw, back
end

struct Context
end

struct J{S,T}
  t::T
end

J{S}(x) where S = J{S,typeof(x)}(x)

Base.show(io::IO, j::J{S}) where S = print(io, "J{$(S.parameters[1])}(...)")

function _forward(ctx::Context, f, args...)
  T = typesof(f, args...)
  (g = _lookup_grad(T)) == nothing &&
    return f(args...), Δ -> error("Undifferentiable function $f")
  meta, forw, _ = g
  nargs = meta.method.nargs
  meta.method.isva && (args = (args[1:nargs-2]...,args[nargs-1:end]))
  y, c = interpret(forw, _forward, ctx, f, args..., sparams = [meta.static_params...])
  return y, J{T}(c)
end

_forward(args...) = _forward(Context(), args...)

function (j::J{T})(Δ) where T
  (g = _lookup_grad(T)) == nothing && return map(_ -> nothing, j.t)
  _, _, back = g
  return interpret(back, j.t, Δ)
end

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
