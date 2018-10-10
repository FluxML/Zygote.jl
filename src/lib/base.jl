using Base: @get!

@nograd readline

@grad copy(x) = copy(x), ȳ -> (ȳ,)

grad_mut(d::AbstractDict) = Dict()

# TODO perhaps look up mutable gradients in `forward`
function accum(a::AbstractDict, b::AbstractDict)
  @assert a === b
  return a
end

@grad function getindex(d::AbstractDict, k)
  d[k], function (Δ)
    grad = grad_mut(__context__, d)
    grad[k] = accum(get(grad, k, nothing), Δ)
    return (grad, nothing)
  end
end

@grad! function setindex!(d::AbstractDict, v, k)
  setindex!(d, v, k), function (Δ)
    (nothing, get(grad_mut(__context__, d), k, nothing), nothing)
  end
end
