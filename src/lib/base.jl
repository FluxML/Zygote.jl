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

function _forward(cx::Context, ::typeof(setindex!), d::AbstractDict, v, k)
  setindex!(d, v, k), function (Δ)
    (nothing, nothing, get(grad_mut(cx, d), k, nothing), nothing)
  end
end
