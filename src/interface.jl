using Base: RefValue

grad(x::Real) = zero(x)
grad(x::Integer) = zero(float(x))

grad(x) = nothing

deref(x) = x
deref(x::RefValue) = x[]
gradref(x) = RefValue(grad(x))

accum!(r::RefValue, x) = (r.x += deref(x))

backprop(J, Δx) = J(Δx)

function backprop(J, Δx::RefValue)
  Δy = J(Δx.x)
  Δx.x = grad(Δx.x)
  return Δy
end

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
