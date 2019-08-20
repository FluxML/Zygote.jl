using Zygote, Test
using Zygote: bufferfrom

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

function tasks4(x)
  ch = Channel(Inf)
  @sync begin
    t = @spawn put!(ch, x^2)
    take!(ch)
  end
end

@test gradient(tasks4, 5) == (10,)

function tasks5(xs)
  n = length(xs)
  chunks = view(xs, 1:n÷2), view(xs, n÷2+1:n)
  p = bufferfrom([0.0, 0.0])
  @sync begin
    for i = 1:2
      @spawn begin
        s = zero(eltype(chunks[i]))
        for j = 1:length(chunks[i])
          s += chunks[i][j]
        end
        p[i] = s
      end
    end
  end
  return p[1]+p[2]
end

Zygote.gradient(tasks5, [1, 2, 3, 4]) == ([1, 1, 1, 1],)
