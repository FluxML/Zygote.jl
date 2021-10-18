using InteractiveUtils
using InteractiveUtils: typesof
using Core: Typeof
import Base: copy!, IdSet
import Base.Broadcast: broadcasted, materialize!

# Internal container used to track accumulated gradients of mutable types (including params).
# Type param I ∈ (true, false) indicates whether implicit params are in use.
# By default, this should be false unless pullback(f, ::Params) is called.
mutable struct Context{I} <: AContext
  cache::Union{IdDict{Any,Any},Nothing}
end

Context() = Context{false}(nothing)

cache(cx::Context) = cx.cache === nothing ? (cx.cache = IdDict()) : cx.cache

struct Pullback{S,T}
  t::T
end

Pullback{S}(x) where S = Pullback{S,typeof(x)}(x)

struct CompileError
  T
  e
end

function Base.showerror(io::IO, e::CompileError)
  print(io, "Compiling $(e.T): ")
  showerror(io, e.e)
end

# interface2.jl

# Wrappers
_pullback(f, args...) = _pullback(Context(), f, args...)

tailmemaybe(::Nothing) = nothing
tailmemaybe(x::Tuple) = unthunk_tangent(Base.tail(x))

# unthunking is essentially an identity operation on a lazy value, but
# `@adjoint unthunk_tangent(x) = unthunk_tangent(x), ȳ -> (ȳ,)` is not enough to make
# nested AD work, so define
@adjoint tailmemaybe(xs::Tuple) = tailmemaybe(xs), x̄s -> ((nothing, x̄s...),)


"""
    pullback(f, args...)
    pullback(f, ::Params)

Returns the value of the function `f` and a back-propagator function,
which can be called to obtain a tuple containing `∂f/∂x` for each argument `x`,
the derivative (for scalar `x`) or gradient.

```julia
y, back = pullback(f, args...)
∇ = back(seed)
```

`back` must be called with a start value `seed` matching the output of `f(args...)`.
If `f(args...)` returns a number, `seed` should be a number.
If `f(args...)` returns an array, `seed` should be an equally-sized array.

See also [`withgradient`](@ref) to obtain the value and gradients in one call,
and [`gradient`](@ref) for obtaining just the gradients.

```jldoctest; setup=:(using Zygote)
julia> y, back = pullback(*, 2.0, 3.0, 5.0);

julia> y
30.0

julia> back(1.0)
(15.0, 10.0, 6.0)

julia> back(2.0)
(30.0, 20.0, 12.0)

julia> y, back = pullback(x -> [x, x], 1.0);

julia> y
2-element Vector{Float64}:
 1.0
 1.0

julia> back([1.0, 1.0])
(2.0,)

julia> back([2.0, nothing])
(2.0,)
```
"""
@inline pullback(f, args...) = pullback(f, Context(), args...)
function pullback(f, cx::AContext, args...)
  y, back = _pullback(cx, f, args...)
  y, Δ -> tailmemaybe(back(Δ))
end
function pullback(cx::Context, f, args...)
  ChainRulesCore.ignore_derivatives() do
    @warn """
    Incorrect argument order for pullback, please use:

      pullback(f, __context__::Context, args)

    instead of:

      pullback(__context__::Context, f, args)

    This is usually caused by a call to pullback in a higher-order @adjoint.
    The above warning will become an error in Zygote 0.7.
    """
  end
  return pullback(f, cx, args...)
end

sensitivity(y::Number) = one(y)
sensitivity(y::Complex) = error("Output is complex, so the gradient is not defined.")
sensitivity(y::AbstractArray) = error("Output is an array, so the gradient is not defined. Perhaps you wanted jacobian.")
sensitivity(y) = error("Output should be scalar; gradients are not defined for output $(repr(y))")

# Preserves output as tuple when gradients are collapsed
_project_all(::NTuple{N}, ::Nothing) where {N} = ntuple(_ -> nothing, N)
_project_all(x::Tuple, dx::Tuple) = map(_project, x, dx)

"""
    gradient(f, args...)

Returns a tuple containing `∂f/∂x` for each argument `x`,
the derivative (for scalar `x`) or the gradient.
If no gradient is defined, `∂f/∂x` will be `nothing`.

`f(args...)` must be a real number, see [`jacobian`](@ref) for array output.

See also [`withgradient`](@ref) to keep the value `f(args...)`,
and [`pullback`](@ref) for value and back-propagator.

```jldoctest; setup=:(using Zygote)
julia> gradient(*, 2.0, 3.0, 5.0)
(15.0, 10.0, 6.0)

julia> gradient(x -> sum(abs2,x), [7.0, 11.0, 13.0])
([14.0, 22.0, 26.0],)

julia> gradient([7, 11], 0, 1) do x, y, d
         p = size(x, d)
         sum(x.^p .+ y)
       end
([14.0, 22.0], 2.0, nothing)
```
"""
function gradient(f, args...)
  y, back = pullback(f, args...)
  grad = back(sensitivity(y))
  return _project_all(args, grad)
end

# Base.adjoint(f::Function) = x -> gradient(f, x)[1]  # piracy!
Base.adjoint(f::Function) = x -> begin  # still piracy! avoids projection for legacy reasons
  y, back = pullback(f, x)
  back(sensitivity(y))[1]
end

"""
    withgradient(f, args...)
    withgradient(f, ::Params)

Returns both the value of the function and the [`gradient`](@ref),
as a named tuple.

```jldoctest; setup=:(using Zygote)
julia> y, ∇ = withgradient(/, 1, 2)
(val = 0.5, grad = (0.5, -0.25))

julia> ∇ == gradient(/, 1, 2)
true
```

Allows you to capture auxillary outputs, in addition to the scalar
used by `gradient`. To do this, `f` must return a Tuple or NamedTuple.
Then it calculates `grad = gradient(first∘f, args...)
but returns the whole `val = f(args...)`:

```jldoctest; setup=:(using Zygote)
julia> withgradient([1,2,4]) do x
          z = 1 ./ x
          sum(z), z  # here z is an auxillary output
       end
(val = (1.75, [1.0, 0.5, 0.25]), grad = ([-1.0, -0.25, -0.0625],))

julia> withgradient(3.0, 4.0) do x, y
          (div = x/y, mul = x*y)
       end
(val = (div = 0.75, mul = 12.0), grad = (0.25, -0.1875))
```

Also supports implicit mode:

```jldoctest; setup=:(using Zygote)
julia> w = [3.0];

julia> res = withgradient(() -> sum(abs2, w), Params([w]))
(val = 9.0, grad = Grads(...))

julia> res.grad[w]
1-element Vector{Float64}:
 6.0
```
"""
function withgradient(f, args...)
  y, back = pullback(f, args...)
  grad = if y isa Tuple
    dy = (sensitivity(first(y)), map(_ -> nothing, Base.tail(y))...)
    back(dy)
  elseif y isa NamedTuple
    dy = (sensitivity(first(y)), map(_ -> nothing, Base.tail(y))...)
    back(NamedTuple{propertynames(y), typeof(dy)}(dy))
  else
    back(sensitivity(y))
  end
  results = _project_all(args, grad)
  (val=y, grad=results)
end

# Param-style wrappers

"""
    gradient(() -> loss(), ps::Params) -> Grads

Gradient with implicit parameters. Takes a zero-argument function,
and returns a dictionary-like container, whose keys are arrays `x in ps`.

See also [`withgradient`](@ref) to keep the value `loss()`.

```jldoctest; setup=:(using Zygote)
julia> x = [1 2 3; 4 5 6]; y = [7, 8]; z = [1, 10, 100];

julia> g = gradient(Params([x, y])) do
         sum(x .* y .* z')
       end
Grads(...)

julia> g[x]
2×3 Matrix{Float64}:
 7.0  70.0  700.0
 8.0  80.0  800.0

julia> haskey(g, z)  # only x and y are parameters
false
```
"""
gradient

"""
    Params([A, B])

Container for implicit parameters, used when differentiating
a zero-argument function `() -> loss(A, B)` with respect to `A, B`.
"""
struct Params{B <: Buffer}
  order::B
  params::IdSet{Any} # TODO store ids only
end

Params() = Params(Buffer([], false), IdSet())
Params(xs) = Params(Buffer(xs, false), IdSet{Any}(xs))
Params(ps::Params) = ps
Params(xs::Tuple) = Params(collect(xs))

@forward Params.order Base.iterate, Base.length, Base.getindex

Base.in(x, ps::Params) = x in ps.params

Base.map(::typeof(_project), args::Tuple{Params}, grad) = grad  # skip _project in gradient(f, ::Params)

function Base.union!(ps::Params, itrs...)
  foreach(itr -> foreach(x -> push!(ps, x), itr), itrs)
  return ps
end

Base.copy(ps::Params) = union!(Params(), ps)
Base.union(ps::Params, itrs...) = union!(copy(ps), itrs...)
Base.issetequal(ps1::Params, ps2::Params) = issetequal(ps1.params, ps2.params)
Base.issetequal(ps1::Params, x::Base.AbstractSet) = issetequal(ps1.params, x)
Base.issetequal(x::Base.AbstractSet, ps1::Params) = issetequal(x, ps1.params)

function Base.intersect!(ps::Params, itrs...)
  for itr in itrs
    for x in collect(ps)
      x ∉ itr && delete!(ps, x)
    end
  end
  return ps
end

Base.intersect(ps::Params, itrs...) = intersect!(copy(ps), itrs...)

function Base.push!(ps::Params, x)
  if !(x in ps.params)
    push!(ps.order, x)
    push!(ps.params, x)
  end
  return ps
end

Base.push!(ps::Params, x...) = (foreach(x -> push!(ps, x), x); ps)

function Base.delete!(ps::Params, x)
  if x in ps.params
    delete!(ps.params, x)
    i = findfirst(y -> y === x, ps.order)
    deleteat!(ps.order, i)
  end
  return ps
end

Base.Broadcast.broadcasted(f, ps::Params) = broadcasted(f, ps.order)

@adjoint function Broadcast.broadcasted(f::Function, ps::Params)
  f.(ps), _ -> throw(ArgumentError("Zygote.Params does not support broadcasting within gradients, try iteration `for p in ps`"))
end

Base.:(==)(x::Params, y::Params) = x.order.data == y.order.data

function Base.show(io::IO, ps::Params)
  print(io, "Params([")
  join(io, ps.order, ", ")
  print(io, "])")
end


"""
    copy!(ps::Params, x::AbstractVector)
    copy!(x::AbstractVector, ps::Params)

Copies the content of array `x` into the parameters `ps` or viceversa.
The length of `x` has to be equal to the sum of the lengths
of all parameters.
"""
function copy!(ps::Params, x::AbstractVector)
  @assert length(x) == sum(length(p) for p in ps)
  i = 0
  for p in ps
    p .= reshape(x[i+1:i+length(p)], size(p))
    i += length(p)
  end
  ps
end

function copy!(x::AbstractVector, ps::Params)
  @assert length(x) == sum(length(p) for p in ps)
  i = 0
  for p in ps
    x[i+1:i+length(p)] .= vec(p)
    i += length(p)
  end
  x
end

"""
    Grads(...)

Dictionary-like container returned when taking gradients with
respect to implicit parameters. For an array `W`, appearing
within `Params([W, A, B...])`, the gradient is `g[W]`.
"""
struct Grads
  grads::IdDict{Any,Any}
  params::Params
end

Base.show(io::IO, ps::Grads) = print(io, "Grads(...)")

@forward Grads.grads  Base.setindex!
@forward Grads.params  Base.length

const ADictOrGrads = Union{AbstractDict, Grads}

# Dictionary interface.
# Don't use the IdDict directly since it may contain some spurious pairs.
Base.haskey(gs::Grads, x) = x ∈ gs.params
Base.keys(gs::Grads) = gs.params
Base.values(gs::Grads) = (gs.grads[p] for p in gs.params)

function Base.iterate(gs::Grads, state...)
  res = iterate(gs.params, state...)
  isnothing(res) && return nothing
  p, next_state = res
  return gs[p], next_state
end

function Base.getindex(gs::Grads, x)
  isbits(x) && error("Only reference types can be differentiated with `Params`.")
  return gs.grads[x]
end

"""
    copy!(gs::Grads, x::AbstractVector)
    copy!(x::AbstractVector, gs::Grads)

Copies the content of array `x` into the gradient object `gs` or vice versa. The
length of `x` has to be equal to the sum of the lengths of all gradients.
"""
function copy!(gs::Grads, x::AbstractVector)
  i = 0
  for p in gs.params
    gs[p] .= reshape(x[i+1:i+length(p)], size(p))
    i += length(p)
  end
  gs
end

function copy!(x::AbstractVector,  gs::Grads)
  i = 0
  for p in gs.params
    x[i+1:i+length(p)] .= vec(gs[p])
    i += length(p)
  end
  x
end

function Base.merge!(gs_dst::Grads, gs_srcs::Grads...)
  for gs_src in gs_srcs
    union!(gs_dst.params, gs_src.params)
    merge!(gs_dst.grads, gs_src.grads)
  end
  gs_dst
end

function Base.copy(gs::Grads)
  gs_new = Grads(IdDict(), gs.params)
  merge!(gs_new, gs)
end

broadcasted(f, gs::Grads, gss::ADictOrGrads...) = map(f, gs, gss...)

broadcasted(f, a::Numeric, gs::Grads) = map(x -> f(a, x), gs)
broadcasted(f, gs::Grads, a::Numeric) = map(x -> f(x, a), gs)

function materialize!(gs1::Grads, gs2::Grads)
  issetequal(gs1.params, gs2.params) ||
    throw(ArgumentError("Expected Grads objects with the same Params."))
  for p in gs1.params
    gs1[p] = gs2[p]
  end
  return gs1
end


function Base.map(f, gs1::Grads, gss::ADictOrGrads...)
  gsout = Grads(IdDict{Any,Any}(), Params(gs1.params))
  return map!(f, gsout, gs1, gss...)
end

function Base.map!(f, gsout::Grads, gss::ADictOrGrads...)
  all(issetequal(gsout.params, keys(gs)) for gs in gss) ||
    throw(ArgumentError("map! expects Grads objects with the same Params."))
  for p in gsout.params
    gsout[p] = f((_getformap(gs, p) for gs in gss)...)
  end
  return gsout
end

function _getformap(gs, p)
  g = gs[p]
  isnothing(g) ? fill!(similar(p), 0) : g
end

function pullback(f, ps::Params)
  cx = Context{true}(nothing)
  y, back = _pullback(cx, f)
  y, function (Δ)
    for p in ps
      cache(cx)[p] = nothing
    end
    back(Δ)
    Grads(unthunk_tangent(cx.cache), ps) # TODO make a copy
  end
end

# No conversion required here
_project_all(_, dx::Grads) = dx

# Code Reflection

function code_ir(f, T)
  m = meta(Tuple{Typeof(f),T.parameters...})
  return IR(m)
end

function code_irm(ex)
  isexpr(ex, :call) || error("@code_ir f(args...)")
  f, args = ex.args[1], ex.args[2:end]
  :(code_ir($(esc(f)), typesof($(esc.(args)...))))
end

macro code_ir(ex)
  code_irm(ex)
end

macro code_adjoint(ex)
  :(Adjoint($(code_irm(ex)), varargs = varargs($(esc(:($InteractiveUtils.@which $ex))), length(($(esc.(ex.args)...),)))))
end
