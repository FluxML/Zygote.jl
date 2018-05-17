macro code_grad(ex)
  :(grad_ir($(code_irm(ex))))
end

isprimitive(f) = f isa Core.Builtin || f isa Core.IntrinsicFunction

function unsplat(f, args...)
  m = which(f, typesof(args...))
  m.isva || return args
  (args[1:m.nargs-2]...,args[m.nargs-1:end])
end

function _forward(f, args...)
  isprimitive(f) && return (f(args...), Δ -> map(_ -> nothing, args))
  ir = code_ir(f, typesof(args...))
  forw, back = stacks!(grad_ir(ir))
  y, J = interpret(forw, f, unsplat(f, args...)...)
  return y, Δ -> interpret(back, J, Δ)
end

function forward(f, args...)
  y, back = _forward(f, args...)
  y, Δ -> Base.tail(back(Δ))
end

function gradient(f, args...)
  y, J = forward(f, args...)
  y isa Real || error("Function output is not scalar")
  return J(1)
end
