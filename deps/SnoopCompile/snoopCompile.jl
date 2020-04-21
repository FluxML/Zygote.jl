using SnoopCompile

@snoopi_bot "Zygote" begin
  using Zygote
  using InteractiveUtils

  function pow(x, n)
    r = 1
    while n > 0
      n -= 1
      r *= x
    end
    return r
  end

  Zygote.@code_adjoint pow(2, 3)

  Zygote.@code_adjoint ((x, y) -> sum(x.*y))([1,2,3],[4,5,6])

  Zygote.gradient(pow, 2, 3)
  Zygote.gradient((x, y) -> sum(x.*y), [1, 2, 3], [4, 5, 6])

  Zygote.gradient(x -> sum(Float32.(x)), [1.0])
end
