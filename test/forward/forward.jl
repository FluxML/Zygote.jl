using Zygote, Test
using NNlib: relu

D(f, x) = pushforward(f, x)(1)

@test D(x -> sin(cos(x)), 0.5) == -cos(cos(0.5))*sin(0.5)

@test D(x -> D(cos, x), 0.5) == -cos(0.5)

@test D(x -> x*D(y -> x*y, 1), 4) == 8

function pow(x, n)
  r = 1
  while n > 0
    n -= 1
    r *= x
  end
  return r
end

@test D(x -> pow(x, 3), 2) == 12

@test D(1) do x
  f(y) = x = x*y
  D(f, 1)
  D(f, 1)
end == 1

@test D(x -> D(y -> x = y, x)*x, 1) == 1

@test D(1) do x
  D(2) do y
    D(3) do z
      x = z * y
    end
  end
  x
end == 0
