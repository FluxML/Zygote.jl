using Zygote, Test

function tasks1(x)
  ch = Channel(Inf)
  put!(ch, x^2) + take!(ch)
end

@test gradient(tasks1, 5) == (20,)

function tasks2(x)
  ch = Channel(0)
  t = @async put!(ch, x^2)
  y = take!(ch)
  wait(t)
  return y
end

@test gradient(tasks2, 5) == (10,)

function tasks3(x)
  ch = Channel(0)
  @sync begin
    @async put!(ch, x^2)
    take!(ch)
  end
end

@test gradient(tasks3, 5) == (10,)

tasks4(x) = fetch(@async x^2)

@test gradient(tasks4, 5) == (10,)

VERSION > v"1.3-" && include("threads.jl")

@test Zygote.pullback(Array, [1f0])[1] == [1f0]

@testset "Dicts" begin

  d = Dict()
  x = [5.0]
  f = () -> get!(() -> sum(x .* x), d, x)

  grads = gradient(f, Params((x,)))
  @test d == Dict(x => 25)
  @test grads[d] == Dict()
  @test grads[x] == [10]

  # This time it is already in the dict
  grads = gradient(f, Params((x,)))
  @test d == Dict(x => 25)
  @test grads[d] == Dict(x => 1) # Not sure why tbh...
  @test grads[x] == nothing

  # Test (non-mutating) get. Same result as get! when x is in d
  f = () -> get(() -> sum(x .* x), d, x)

  grads = gradient(f, Params((x,)))
  @test d == Dict(x => 25)
  @test grads[d] == Dict(x => 1) # Not sure why tbh...
  @test grads[x] == nothing

  delete!(d, x)

  grads = gradient(f, Params((x,)))
  @test d == Dict() # Nothing added with get
  @test d ∉ keys(grads.grads)
  @test grads[x] == [10]
end
