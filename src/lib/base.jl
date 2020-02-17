using Base: @get!

@nograd readline, Base.gc_num, Base.time_ns

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

@nograd haskey

# Channels

@nograd Channel, schedule

grad_mut(ch::Channel) = Channel(ch.sz_max)

@adjoint! function put!(ch::Channel, x)
  put!(ch, x), function (ȳ)
    x̄ = take!(grad_mut(__context__, ch))
    (nothing, accum(x̄, ȳ), nothing)
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
  @static if VERSION > v"1.3-"
    t̄.sticky = t.sticky
  end
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

# Make @sync work
# Safe as long as other kinds of mutation are disallowed
@adjoint push!(refs::Vector{Any}, t::Task) = push!(refs, t), _ -> nothing
