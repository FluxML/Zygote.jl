using Base: @get!

@nograd readline

@adjoint copy(x::AbstractArray) = copy(x), ȳ -> (ȳ,)

grad_mut(d::AbstractDict) = Dict()

# TODO perhaps look up mutable gradients in `forward`
function accum(a::AbstractDict, b::AbstractDict)
  @assert a === b
  return a
end

@adjoint function getindex(d::AbstractDict, k)
  d[k], function (Δ)
    grad = grad_mut(__context__, d)
    grad[k] = accum(get(grad, k, nothing), Δ)
    return (grad, nothing)
  end
end

@adjoint! function setindex!(d::AbstractDict, v, k)
  setindex!(d, v, k), function (Δ)
    (nothing, get(grad_mut(__context__, d), k, nothing), nothing)
  end
end
