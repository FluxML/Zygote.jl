# Gradient of AD stacks

grad_mut(::AbstractVector) = []

@adjoint! function _push!(a::Vector, x)
  _push!(a, x), function (y)
    dstk = grad_mut(__context__, a)
    return (nothing, pop!(dstk))
  end
end

@adjoint! function pop!(stk::Stack)
  pop!(stk), function (Δ)
    dstk = grad_mut(__context__, stk.data)
    push!(dstk, Δ)
    return
  end
end

# Dictionaries

grad_mut(d::AbstractDict) = Dict()
grad_mut(d::IdDict) = IdDict()

# TODO perhaps look up mutable gradients in `pullback`
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
  setindex!(d, v, k), function (_)
    Δ = get(grad_mut(__context__, d), k, nothing)
    delete!(grad_mut(__context__, d), k)
    (nothing, Δ, nothing)
  end
end

# Channels

@nograd Channel

grad_mut(ch::Channel) = Channel(ch.sz_max)

@adjoint! function put!(ch::Channel, x)
  put!(ch, x), function (ȳ)
    x̄ = grad_mut(__context__, ch)
    dx = isopen(x̄) ? take!(x̄) : nothing
    (nothing, accum(dx, ȳ), nothing)
  end
end

@adjoint! function take!(ch::Channel)
  take!(ch), function (x̄)
    put!(grad_mut(__context__, ch), x̄)
    return
  end
end

@adjoint! function Task(f)
  t = Task(f)
  t.code = function ()
    y, back = _pullback(__context__, f)
    cache(__context__)[t] = Task(back)
    return y
  end
  t, _ -> fetch(cache(__context__)[t])
end

function runadjoint(cx, t, ȳ = nothing)
  t̄ = cache(cx)[t]
  f = t̄.code
  t̄.code = () -> f(ȳ)
  t̄.sticky = t.sticky
  schedule(t̄)
end

@adjoint! function wait(t::Task)
  wait(t), _ -> (runadjoint(__context__, t); nothing)
end

@adjoint! function fetch(t::Task)
  fetch(t), ȳ -> (runadjoint(__context__, t, ȳ); nothing)
end

@adjoint! function Base.sync_end(refs)
  Base.sync_end(refs), _ -> foreach(t -> runadjoint(__context__, t), refs)
end

# Need to hold onto the references here
@adjoint! function Base.sync_end(ch::Channel)
  ch_copy = grad_mut(__context__, ch)
  dch = grad_mut(ch_copy)
  while !isempty(ch)
    i = take!(ch)
    put!(ch_copy, i)
    put!(dch, i)
  end
  Base.sync_end(ch_copy), _ -> begin
    while !isempty(dch)
      t = take!(dch)
      runadjoint(__context__, t)
    end
  end
end

# Make @sync work
# Safe as long as other kinds of mutation are disallowed
@adjoint push!(refs::Vector{Any}, t::Task) = push!(refs, t), _ -> nothing

# named tuple
@adjoint function pairs(t::NamedTuple{N}) where N
    pairs_namedtuple(dx::NamedTuple) = (dx.data,)
    function pairs_namedtuple(Δ::Dict)
        t0 = map(zero, t)
        for (idx, v) in Δ
            t0 = NamedTuple{N}(Base.setindex((t0...,), v, idx))
        end
        return (t0,)
    end
    return pairs(t), pairs_namedtuple
end

@adjoint function Base.getfield(p::Pair, i::Int)
    function pair_getfield(Δ)
        f, s = i == 1 ? (Δ, zero(p[2])) : (zero(p[1]), Δ)
        return (first=f, second=s), nothing
    end
    return getfield(p, i), pair_getfield
end

@adjoint Base.nameof(x::UnionAll) = nameof(x), _ -> (nothing,)

@nograd typeintersect
