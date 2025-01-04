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

# IdSet (needed for nested AD with implicit params)

grad_mut(::IdSet) = IdSet()

function _pullback(cx::AContext, ::typeof(push!), s::IdSet, @nospecialize(x))
  res = push!(s, x)
  function idset_push!_pullback(_)
    Δ = pop!(grad_mut(cx, s), x, nothing)
    (nothing, Δ, nothing)
  end
  return res, idset_push!_pullback
end

# Dictionaries

grad_mut(d::AbstractDict) = Dict()
grad_mut(d::IdDict) = IdDict()

# TODO perhaps look up mutable gradients in `pullback`
function accum(a::AbstractDict, b::AbstractDict)
  a === b && return a # Mutating case
  return merge(a, b)
end

@adjoint function getindex(d::AbstractDict, k)
  val = d[k]
  function dict_getindex_pullback(Δ)
    accum_param(__context__, val, Δ) === nothing && return
    grad = grad_mut(__context__, d)
    grad[k] = accum(get(grad, k, nothing), Δ)
    return (grad, nothing)
  end
  val, dict_getindex_pullback
end

@adjoint! function setindex!(d::AbstractDict, v, k)
  setindex!(d, v, k), function (_)
    Δ = get(grad_mut(__context__, d), k, nothing)
    delete!(grad_mut(__context__, d), k)
    (nothing, Δ, nothing)
  end
end

# This rule behaves much like the getindex adjoint,
# just with an (internal) ordinal index instead of a key.
function _pullback(cx::AContext, ::typeof(iterate), d::Dict, i)
  iter = iterate(d, i)
  function dict_iterate_pullback(Δ)
    (iter === nothing || Δ === nothing) && return
    k, v = iter[1]
    _, dv = Δ[1]
    accum_param(cx, v, dv) === nothing && return
    grad = grad_mut(cx, d)
    grad[k] = accum(get(grad, k, nothing), dv)
    return (nothing, grad, nothing)
  end
  return iter, dict_iterate_pullback
end

# ...while this one is to avoid duplicating code or differentiating skip_deleted.
# The alternative would be to write a rule for the private _iterate(::Dict, i).
function _pullback(cx::AContext, ::typeof(iterate), d::Dict)
  # Calculation of i is the same used in iterate(::Dict)
  return _pullback(cx, iterate, d, Base.skip_deleted(d, d.idxfloor))
end

function _pullback(cx::AContext, ::typeof(iterate), vi::Base.ValueIterator{<:Dict}, i::Int)
  iter = iterate(vi, i)
  function values_iterate_pullback(Δ)
    (iter === nothing || Δ === nothing) && return
    v, dv = iter[1], Δ[1]
    accum_param(cx, v, dv) === nothing && return
    # Same as vi.dict.keys[i], but without reaching into Dict internals.
    # Iterating the dict instead of keys() is to hit the rules above in nested AD.
    k = iterate(vi.dict, i)[1][1]
    grad = grad_mut(cx, vi.dict)
    grad[k] = accum(get(grad, k, nothing), dv)
    return (nothing, (; dict = grad), nothing)
  end
  return iter, values_iterate_pullback
end

# Channels

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

  pairs_namedtuple_pullback(dx::NamedTuple) = (dx.data,)

  pairs_namedtuple_pullback(dx::Tuple{}) = (NamedTuple(),)

  function pairs_namedtuple_pullback(Δ::Dict)
    t0 = map(zero, t)
    for (idx, v) in Δ
      ii = idx isa Integer ? idx : findfirst(==(idx), keys(t))
      t0 = NamedTuple{N}(Base.setindex((t0...,), v, ii))
    end
    return (t0,)
  end

  return pairs(t), pairs_namedtuple_pullback
end

# For merge between NamedTuple and Dict, we will just convert the Dict to a NamedTuple.
# and then call `pullback`, which should overall be pretty efficient code generated,
# and it avoids trying to AD the problematic generic `merge(::NamedTuple, ::iter)` method which uses `push!`.
function _pullback(cx::AContext, ::typeof(merge), a::NamedTuple, b::Dict{Symbol})
  res, back = _pullback(cx, merge, a, NamedTuple(b))
  return res, back ∘ unthunk_tangent
end

# Keyword arguments pretend to be a Dict, but are secretly wrapping a NamedTuple.
# We can treat them much the same, just with some plumbing to handle the extra `itr` field.
function _pullback(::AContext, ::typeof(getindex),
                   ps::Iterators.Pairs{<:Any,<:Any,<:Any,<:NamedTuple}, k)
  # So we don't close over kwarg values in the pullback
  data = map(_ -> nothing, NamedTuple(ps))
  function kwargs_getindex_pullback(Δ)
    dps = (data = Base.setindex(data, Δ, k), itr = nothing)
    return (nothing, dps, nothing)
  end
  return ps[k], kwargs_getindex_pullback
end

function _pullback(cx::AContext, ::typeof(literal_getindex),
                   ps::Iterators.Pairs{<:Any,<:Any,<:Any,<:NamedTuple}, ::Val{K}) where K
  val, gf_back = _pullback(cx, literal_getfield, NamedTuple(ps), Val(K))
  function kwargs_literal_getindex_pullback(Δ)
    dps = (data = gradindex(gf_back(Δ), 2), itr = nothing)
    return (nothing, dps, nothing)
  end
  return val, kwargs_literal_getindex_pullback
end

# Misc.
@adjoint function Base.getfield(p::Pair, i::Int)
    function pair_getfield_pullback(Δ)
        f, s = i == 1 ? (Δ, nothing) : (nothing, Δ)
        return (first=f, second=s), nothing
    end
    return getfield(p, i), pair_getfield_pullback
end

@adjoint Base.nameof(x::UnionAll) = nameof(x), _ -> (nothing,)

# Base.Fix1 and Base.Fix2: https://github.com/FluxML/Zygote.jl/issues/957
@adjoint function (g::Base.Fix1)(y)
    f = g.f
    x = g.x
    fallback_Fix1(y) = f(x, y)
    return _pullback(__context__, fallback_Fix1, y)
end
@adjoint function (g::Base.Fix2)(y)
    f = g.f
    x = g.x
    fallback_Fix2(y) = f(y, x)
    return _pullback(__context__, fallback_Fix2, y)
end
