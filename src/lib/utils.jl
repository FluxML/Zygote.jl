hook(f, x) = x

@adjoint! hook(f, x) = x, x̄ -> (nothing, f(x̄),)

macro showgrad(x)
  :(hook($(esc(x))) do x̄
      println($"∂($x) = ", repr(x̄))
      x̄
    end)
end

hessian(f, x::AbstractArray) = forward_jacobian(x -> gradient(f, x)[1], x)[2]
