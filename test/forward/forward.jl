using Zygote, Test

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

@test D(x -> abs(x+2im), 1) == gradient(x -> abs(x+2im), 1+0im)[1]
@test real(D(x -> abs(x+2im), 1)) == gradient(x -> abs(x+2im), 1)[1]  # ProjectTo means gradient here is real

@test D(3) do x
  A = zeros(5, 5)
  B = zeros(5, 5)
  A[1, 1] = x
  mul!(B, A, A)
  sum(B)
end == 6
