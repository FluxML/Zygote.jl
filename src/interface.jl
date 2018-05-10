grad(x::Real) = zero(x)
grad(x::Integer) = zero(float(x))
grad(x::Tuple) = grad.(x)

grad(x) = nothing

accum(x, y) = x + y
accum(x, ::Void) = x
accum(::Void, _) = nothing
accum(x::Tuple, y::Tuple) = accum.(x, y)

backprop(J, Δx) = J(Δx)

macro code_grad(ex)
  :(grad_ir($(code_irm(ex))))
end

function ∇(f, args...)
  ir = code_ir(f, typesof(args...))
  forw, back = stacks!(grad_ir(ir))
  y, J = interpret(forw, f, args...)
  return y, function (Δ)
    interpret(back, J, Δ)
  end
end

gradient(f, args...) = ∇(f, args...)[2](1)
