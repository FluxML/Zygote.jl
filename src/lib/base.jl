using Base: @get!

@nograd readline

# Gradient of AD stacks

@adjoint! function _push!(a::Vector, x)
  _push!(a, x), function (y)
    dstk = grad_mut(__context__, a)
    return (nothing, pop!(dstk))
  end
end

@adjoint! function pop!(stk::Stack)
  i = stk.idx
  pop!(stk), function (Δ)
    dstk = grad_mut(__context__, stk.data)
    dstk[i] = Δ
    return
  end
end

# Dictionaries

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
