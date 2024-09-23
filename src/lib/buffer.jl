grad_mut(cx::Context, b::Buffer, ::Type=Union{}) =
  _get!(() -> fill!(similar(b.data, Any), nothing), cache(cx), b)
# S is the eltype we are about to set into the buffer accumulator, so allocte wide enough
grad_mut(cx::Context, b::Buffer{T}, ::Type{S}=Union{}) where {T<:Number, S<:Number} =
  _get!(() -> fill!(similar(b.data, float(promote_type(T, S))), 0), cache(cx), b)

@non_differentiable Buffer(::Any...)

@adjoint function getindex(b::Buffer, i...)
  function getindex_buffer_pullback(Δ)
    grad = grad_mut(__context__, b, eltype(Δ))
    grad[i...] = accum(grad[i...], Δ)
    return
  end
  b[i...], getindex_buffer_pullback
end

@adjoint! function setindex!(b::Buffer, v, i...)
  function setindex!_buffer_pullback(_)
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
  setindex!(b, v, i...), setindex!_buffer_pullback
end

@adjoint! function copyto!(b::Buffer, src::AbstractArray)
  function copyto!_buffer_array_pullback(_)
    grad = grad_mut(__context__, b)
    xs = copy(grad)
    grad .= eltype(grad) <: Number ? zero(eltype(grad)) : nothing
    return (nothing, xs)
  end
  copyto!(b, src), copyto!_buffer_array_pullback
end

@adjoint! function copyto!(b::Buffer, bc::Base.Broadcast.Broadcasted)
  xs, map_pullback = ∇map(__context__, i -> bc[i], eachindex(bc))
  function copyto!_buffer_broadcast_pullback(_)
    grad = grad_mut(__context__, b)
    d, = map_pullback(reshape(first(grad, length(xs)), size(xs)))
    grad .= eltype(grad) <: Number ? zero(eltype(grad)) : nothing
    return (nothing, d.bc)
  end
  copyto!(b, xs), copyto!_buffer_broadcast_pullback
end

function _pullback(cx::AContext, ::typeof(copyto!), b::Buffer, g::Base.Generator)
    xs, collect_pullback = _pullback(cx, collect, g)
    function copyto!_buffer_generator_pullback(_)
        grad = grad_mut(cx, b)
        _, dg = collect_pullback(reshape(first(grad, length(xs)), size(xs)))
        grad .= eltype(grad) <: Number ? zero(eltype(grad)) : nothing
        return (nothing, nothing, dg)
    end
    copyto!(b, xs), copyto!_buffer_generator_pullback
  end

@adjoint! function push!(b::Buffer, x)
  function push!_buffer_pullback(_)
    grad = grad_mut(__context__, b)
    return (nothing, pop!(grad))
  end
  push!(b, x), push!_buffer_pullback
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
