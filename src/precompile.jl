function pow(x, n)
  r = 1
  while n > 0
    n -= 1
    r *= x
  end
  return r
end

# Invoking the compiler outside of the genfuncs appears
# to make the specialised versions visible inside them,
# leading to a significant first-compile speedup.
@code_adjoint pow(2, 3)

@code_adjoint ((x, y) -> sum(x.*y))([1,2,3],[4,5,6])

gradient(pow, 2, 3)
gradient((x, y) -> sum(x.*y), [1, 2, 3], [4, 5, 6])

gradient(x -> sum(Float32.(x)), [1.0])
