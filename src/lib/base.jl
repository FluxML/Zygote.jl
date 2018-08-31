using Base: @get!

@nograd readline

grad_mut(d::AbstractDict) = Dict()

@grad function getindex(d::AbstractDict, k)
  d[k], function (Δ)
    grad = grad_mut(__context__, d)
    grad[k] = accum(get(grad, k, nothing), Δ)
    return
  end
end

@grad! function setindex!(d::AbstractDict, v, k)
  setindex!(d, v, k), function (Δ)
    (nothing, get(grad_mut(__context__, d), k, nothing), nothing)
  end
end
