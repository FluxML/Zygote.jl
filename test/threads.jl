using Zygote, Test
using Zygote: bufferfrom
using Base.Threads: @spawn

function threads1(x)
  ch = Channel(0)
  @sync begin
    @spawn put!(ch, x^2)
    take!(ch)
  end
end

@test gradient(threads1, 5) == (10,)

function threads2(xs)
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

@test gradient(threads2, [1, 2, 3, 4]) == ([1, 1, 1, 1],)
