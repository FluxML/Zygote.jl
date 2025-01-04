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

tasks5(x) = fetch(schedule(Task(() -> x^2)))
@test gradient(tasks5, 5) == (10,)

include("threads.jl")

@test Zygote.pullback(Array, [1f0])[1] == [1f0]

@testset "#300" begin
  t = (rand(2, 2), rand(2, 2))
  ps = Params(t)
  gs = gradient(()->sum(t[1]), ps)
  @test gs[t[1]] == ones(2, 2)
end

struct A594 x::Float64 end
  
@testset "#594" begin
  f(a,v) = a.x + v
  g(A,V) = sum(f.(A,V))
  X = A594.(randn(2))
  Y = randn(2,2)
  ∇ = gradient(g,X,Y)
  @test ∇[1] == [(x = 2.0,); (x = 2.0,)]
  @test vec(∇[1]) == [(x = 2.0,); (x = 2.0,)]
  @test ∇[2] == [1 1; 1 1]
end

@testset "UnionAll Stackoverflow" begin
  struct M{T,B}
    a::T
    b::B
  end

  m, b = Zygote._pullback(Zygote.Context(), nameof, M)
  @test b(m) === nothing
end
