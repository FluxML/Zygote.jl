hook(f, x) = x

@adjoint! hook(f, x) = x, x̄ -> (nothing, f(x̄),)

macro showgrad(x)
  :(hook($(esc(x))) do x̄
      println($"D($x) = ", repr(x̄))
      x̄
    end)
end
