using Flux, BenchmarkTools

include("fakearray.jl")

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 64, relu),
  Dense(64, 32, relu),
  Dense(32, 10, relu),
  softmax)

m = mapleaves(m) do x
  x isa AbstractArray ? FakeArray(x) : x
end

@benchmark $m(x) setup=(x=FakeArray(28^2))

b = @benchmark gradient(x -> sum($m(x)), x) setup=(x=FakeArray(100))

mlp_ops = 48

minimum(b).time / mlp_ops
