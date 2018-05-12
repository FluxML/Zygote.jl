grad(x::Real) = zero(x)
grad(x::Integer) = zero(float(x))
grad(x::Tuple) = grad.(x)

@generated function grad(x)
  (x.mutable || nfields(x) == 0) && return
  Expr(:tuple, [:($f = grad(x.$f)) for f in fieldnames(x)]...)
end

accum(x, y) = x + y
accum(x, ::Void) = x
accum(::Void, _) = nothing
accum(x::Tuple, y::Tuple) = accum.(x, y)

@generated function accum(x::NamedTuple, y::NamedTuple)
  grad(x) = x in fieldnames(y) ? :(y.$x) : :nothing
  Expr(:tuple, [:($f=accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

backprop(J, Δx) = J(Δx)

macro code_grad(ex)
  :(grad_ir($(code_irm(ex))))
end

isprimitive(f) = f isa Core.Builtin || f isa Core.IntrinsicFunction

function _forward(f, args...)
  isprimitive(f) && return (f(args...), Δ -> map(_ -> nothing, args))
  ir = code_ir(f, typesof(args...))
  forw, back = stacks!(grad_ir(ir))
  y, J = interpret(forw, f, args...)
  return y, function (Δ)
    interpret(back, J, Δ)
  end
end

forward(f, args...) = _forward(f, args...)

gradient(f, args...) = forward(f, args...)[2](1)
