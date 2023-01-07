grad_mut(cx::Context, b::Buffer{T}, ::Type{S}=Union{}) where {T, S} =
  get!(() -> fill!(similar(b.data, Any), nothing), cache(cx), b)
grad_mut(cx::Context, b::Buffer{T}, ::Type{S}=Union{}) where {T<:Number, S} =
  get!(() -> fill!(similar(b.data, float(promote_type(T,S))), 0), cache(cx), b)

@non_differentiable Buffer(::Any...)

@adjoint function getindex(b::Buffer, i...)
  b[i...], function (Δ::S) where {S}
    grad = grad_mut(__context__, b, S)
    grad[i...] = accum(grad[i...], Δ)
    return
  end
end

@adjoint! function setindex!(b::Buffer, v, i...)
  setindex!(b, v, i...), function (_)
    grad = grad_mut(__context__, b)
    v̄ = grad[i...]
    zero = eltype(grad) <: Number ? 0 : nothing
    if i isa NTuple{N,Integer} where N
      grad[i...] = zero
    else
      grad[i...] .= zero
    end
    (nothing, v̄, map(_->nothing, i)...)
  end
end

@adjoint! function copyto!(b::Buffer, xs)
  copyto!(b, xs), function (_)
    grad = grad_mut(__context__, b)
    x̄s = copy(grad)
    grad .= eltype(grad) <: Number ? 0 : nothing
    return (nothing, x̄s)
  end
end

@adjoint! function push!(b::Buffer, x)
  push!(b, x), function (y)
    grad = grad_mut(__context__, b)
    return (nothing, pop!(grad))
  end
end

_pullback(cx::AContext, ::typeof(Broadcast.materialize!), b::Buffer, x::AbstractArray) =
  _pullback(cx, copyto!, b, x)

@adjoint function copy(b::Buffer)
  res = copy(b)

  function copy_sensitivity(b̄)
    grad_mut(__context__, b, eltype(b̄))[:] .= vec(b̄)
    return
  end

  function copy_sensitivity(b̄::Union{Tuple,AbstractVector{T}}) where {T<:AbstractArray}
    grad_mut(__context__, b)[:] .= b̄
    return
  end

  return res, copy_sensitivity
end
