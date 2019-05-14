struct Key
  id::UInt64
  Key(x) = new(objectid(x))
end

# Shaves some time on dict lookups (which is all we use this for).
Base.hash(k::Key) = k.id

mutable struct Context
  cache::Union{Dict{Key,Any},Nothing}
  globals::Union{Dict{GlobalRef,Any},Nothing}
end

Context() = Context(nothing, nothing)

cache(cx::Context) = cx.cache === nothing ? (cx.cache = Dict{Key,Any}()) : cx.cache
globals(cx::Context) = cx.globals === nothing ? (cx.globals = Dict{GlobalRef,Any}()) : cx.globals

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

_forward(f, args...) = _forward(Context(), f, args...)

tailmemaybe(::Nothing) = nothing
tailmemaybe(x::Tuple) = Base.tail(x)

function forward(f, args...)
  y, back = _forward(f, args...)
  y, Δ -> tailmemaybe(back(Δ))
end

sensitivity(y::Number) = one(y)
sensitivity(y::Complex) = error("Output is complex, so the gradient is not defined.")
sensitivity(y) = error("Output should be scalar; gradients are not defined for output $y")

function gradient(f, args...)
  y, back = forward(f, args...)
  return back(sensitivity(y))
end

Base.adjoint(f::Function) = x -> gradient(f, x)[1]

# Param-style wrappers

# TODO store ids only
struct Params
  order::Vector{Any}
  params::IdSet{Any}
  Params() = new([], IdSet())
end

@forward Params.order Base.iterate, Base.length

function Base.push!(ps::Params, x)
  if !(x in ps.params)
    push!(ps.order, x)
    push!(ps.params, x)
  end
  return ps
end

Base.push!(ps::Params, x...) = (foreach(x -> push!(ps, x), x); ps)

Params(xs) = push!(Params(), xs...)

function Base.show(io::IO, ps::Params)
  print(io, "Params([")
  join(io, ps.order, ", ")
  print(io, "])")
end

struct Grads
  grads::Dict{Key,Any}
end

Base.show(io::IO, ps::Grads) = print(io, "Grads(...)")

function Base.getindex(gs::Grads, x)
  isbits(x) && error("Only reference types can be differentiated with `Params`.")
  return gs.grads[Key(x)]
end

Base.haskey(gs::Grads, x) = haskey(gs.grads, Key(x))

function forward(f, ps::Params)
  cx = Context()
  y, back = _forward(cx, f)
  y, function (Δ)
    for p in ps
      cache(cx)[Key(p)] = ismutvalue(p) ? grad_mut(p) : nothing
    end
    back(Δ)
    Grads(cx.cache) # TODO make a copy
  end
end

# Code Reflection

using InteractiveUtils
using InteractiveUtils: typesof
using Core: Typeof

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
