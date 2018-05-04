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
