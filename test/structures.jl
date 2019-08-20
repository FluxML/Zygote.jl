using Zygote, Test

function f(x)
  ch = Channel(Inf)
  put!(ch, x^2) + take!(ch)
end

@test gradient(f, 5) == (20,)
