using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  # control flow makes the pullback use stacks
  function pow(x, n)
    r = one(x)
    while n > 0
      n -= 1
      r *= x
    end
    return r
  end
  @compile_workload begin
    gradient(pow, 2.0, 3)
    for T in (Float32, Float64)
      x, b, W = T[1, 2, 3], T[1, 2], T[1 2 3; 4 5 6]
      gradient(x -> sum(abs2, x), x)
      gradient((W, b, x) -> sum(tanh.(W * x .+ b)), W, b, x)
      withgradient(m -> sum(m.W * x .+ m.b), (; W, b))
    end
  end
end
