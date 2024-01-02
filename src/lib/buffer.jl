grad_mut(cx::Context, b::Buffer, ::Type=Union{}) =
  _get!(() -> fill!(similar(b.data, Any), nothing), cache(cx), b)
# S is the eltype we are about to set into the buffer accumulator, so allocte wide enough
grad_mut(cx::Context, b::Buffer{T}, ::Type{S}=Union{}) where {T<:Number, S<:Number} =
  _get!(() -> fill!(similar(b.data, float(promote_type(T, S))), 0), cache(cx), b)

@non_differentiable Buffer(::Any...)

@adjoint function getindex(b::Buffer, i...)
  b[i...], function (Δ)
    grad = grad_mut(__context__, b, eltype(Δ))
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


@adjoint! function push!(b::Buffer, x)
  push!(b, x), function (y)
    grad = grad_mut(__context__, b)
    return (nothing, pop!(grad))
  end
end


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

Base.BroadcastStyle(::Type{Buffer{T,A}}) where {T,A} = Base.BroadcastStyle(A)

@non_differentiable Base.Broadcast.Broadcasted(::Nothing)

function _pullback(cx::AContext, ::typeof(copyto!), b::Buffer, bc::Base.Broadcast.Broadcasted)
  xs, map_pullback = ∇map(cx, i -> bc[i], eachindex(bc))
  copyto!(b, xs), function (_)
    grad = grad_mut(cx, b)
    # ys = copy(grad)
    d, = map_pullback(reshape(first(grad, length(xs)), size(xs)))
    return (nothing, nothing, d.bc)
  end
end
