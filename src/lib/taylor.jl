using TaylorSeries

@adjoint deepcopy(x::Taylor1) = error("deepcopy not implemented")

taylorshift(x, n) = x + Taylor1(typeof(x), n)

derivatives(x::Taylor1, n) = accumulate((x, n) -> derivative(x), 1:n, init = x)
derivatives(x, n) = [zero(x) for i = 1:n]

function derive(x, ns)
  ds = derivatives(x, maximum(ns))
  map(n -> constant_term(ds[n]), ns)
end

"""
    taylordiff(f, x, n = 1)

Return the derivatives of `f(x)` with respect to a scalar `x` at order `n`,
using taylor series expansions.

    julia> taylordiff(x -> [x^2, sin(x)], 5, 2)
    2-element Array{Float64,1}:
     2.0
     0.9589242746631385

Multiple orders may be given at once, and derivatives for each order will be
returned separately.

    julia> taylordiff(x -> [x^2, sin(x)], 5, (1, 2, 3))
    ([10.0, 0.283662], [2.0, 0.958924], [0.0, -0.283662])
"""
function taylordiff(f, x::Real, n = 1)
  N = maximum(n)
  x = taylorshift(x, N)
  y = f(x)::Union{Number,AbstractArray{<:Number}}
  ds = derivatives.(y, N)
  return map(n -> map(d -> constant_term(d[n]), ds), n)
end
