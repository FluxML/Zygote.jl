using Zygote, Base.Test
using Zygote: gradient

function ngradient(f, xs::AbstractArray...)
  grads = zeros.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(broadcast(sin, f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(dims)...)

@testset "Gradients" begin

@test gradtest(*, (2,5), 5)

@test gradtest(x -> sum(x, (2, 3)), rand(3,4,5))

end
