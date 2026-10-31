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
  scaled(x; s = 2) = s * sum(abs2, x)
  function tuplesum(t)
    s = zero(first(t))
    for a in t
      s += a
    end
    return s
  end
  # `y` is a Union{Float64, Int}
  unionbranch(x) = (y = x > 0 ? x : 0; y^2)
  @compile_workload begin
    gradient(pow, 2.0, 3)
    gradient(x -> scaled(x; s = 3), [1.0, 2.0, 3.0])
    gradient(tuplesum, (1.0, 2.0, 3.0))
    gradient(unionbranch, 2.0)
    gradient(x -> (p = Pair(x, 2x); p.first * p.second), 2.0)
    gradient(v -> sum(sum, v), [[1.0, 2.0], [3.0]])
    gradient(d -> d[:a] * d[:b], Dict(:a => 1.0, :b => 2.0))
    gradient(t -> t[1] * t[2] + t[3], (1.0, 2.0, 3.0))
    for T in (Float32, Float64)
      x, b, W = T[1, 2, 3], T[1, 2], T[1 2 3; 4 5 6]
      gradient(x -> sum(abs2, x), x)
      gradient((W, b, x) -> sum(tanh.(W * x .+ b)), W, b, x)
      withgradient(m -> sum(m.W * x .+ m.b), (; W, b))
    end
  end
end
