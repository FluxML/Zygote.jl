@test_deprecated dropgrad(1)
@test_deprecated ignore(1)
@test_deprecated Zygote.@ignore x=1

@test gradient(x -> Zygote.ignore(() -> x*x), 1) == (nothing,)
@test gradient(x -> Zygote.@ignore(x*x), 1) == (nothing,)
@test gradient(1) do x
  y = Zygote.@ignore x
  x * y
end == (1,)
