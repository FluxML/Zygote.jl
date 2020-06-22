using InteractiveUtils
using InteractiveUtils: typesof
using Core: Typeof

@static if VERSION >= v"1.1"
  import Base: copy!
else
  import Future: copy!
end

mutable struct Context <: AContext
  cache::Union{IdDict{Any,Any},Nothing}
end

Context() = Context(nothing)

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
tailmemaybe(x::Tuple) = Base.tail(x)

function pullback(f, args...)
  y, back = _pullback(f, args...)
  y, Δ -> tailmemaybe(back(Δ))
end

sensitivity(y::Number) = one(y)
sensitivity(y::Complex) = error("Output is complex, so the gradient is not defined.")
sensitivity(y) = error("Output should be scalar; gradients are not defined for output $(repr(y))")

function gradient(f, args...)
  y, back = pullback(f, args...)
  return back(sensitivity(y))
end

Base.adjoint(f::Function) = x -> gradient(f, x)[1]

# Param-style wrappers

# TODO store ids only
struct Params
  order::Buffer{Any, Vector{Any}}
  params::IdSet{Any}
  Params() = new(Buffer([], false), IdSet())
end

@forward Params.order Base.iterate, Base.length, Base.getindex

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

Params(xs) = push!(Params(), xs...)

Base.Broadcast.broadcasted(f, ps::Params) = broadcasted(f, ps.order)

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
  ps
end


struct Grads
  grads::IdDict{Any,Any}
  params::Params
end

Base.show(io::IO, ps::Grads) = print(io, "Grads(...)")

@forward Grads.grads Base.getindex, Base.haskey

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
  x
end

function copy!(x::AbstractVector,  gs::Grads)
  i = 0
  for p in gs.params
      x[i+1:i+length(p)] .= vec(gs[p])
      i += length(p)
  end
  x
end

function pullback(f, ps::Params)
  cx = Context()
  y, back = _pullback(cx, f)
  y, function (Δ)
    for p in ps
      cache(cx)[p] = nothing
    end
    back(Δ)
    Grads(cx.cache, ps) # TODO make a copy
  end
end

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
