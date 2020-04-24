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

@testset "#300" begin
  t = (rand(2, 2), rand(2, 2))
  ps = Params(t)
  gs = gradient(()->sum(t[1]), ps)
  @test gs[t[1]] == ones(2, 2)
end

@testset "#594" begin
  struct A x::Float64 end
  f(a,v) = a.x + v
  g(X,Y) = sum(f.(X,Y))
  X = A.(randn(2))
  Y = randn(2,2)
  ∇ = gradient(g,X,Y)
  @test ∇[1] == [(x = 2.0,); (x = 2.0,)]
  @test ∇[2] == [1 1; 1 1]
end

@testset "#594 2" begin
  struct B x::Float64; y::Float64 end
  f(a,v) = a.x + v
  g(X,Y) = sum(f.(X,Y))
  X = B.(randn(2),randn(2))
  Y = randn(2,2)
  ∇ = gradient(g,X,Y)
  @test ∇[1] == [(x=2.0, y=nothing); (x=2.0, y=nothing)]
  @test ∇[2] == [1 1; 1 1]
end
