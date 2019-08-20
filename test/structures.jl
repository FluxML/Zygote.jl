using Zygote, Test

function f(x)
  ch = Channel(Inf)
  put!(ch, x^2) + take!(ch)
end

@test gradient(f, 5) == (20,)

function f(x)
  ch = Channel(0)
  t = @async put!(ch, x^2)
  y = take!(ch)
  wait(t)
  return y
end

@test gradient(f, 5) == (10,)

function f(x)
  ch = Channel(0)
  @sync begin
    @async put!(ch, x^2)
    take!(ch)
  end
end

@test gradient(f, 5) == (10,)
