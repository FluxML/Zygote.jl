struct Key
  id::UInt64
  Key(x) = new(objectid(x))
end

# Shaves some time on dict lookups (which is all we use this for).
Base.hash(k::Key) = k.id

mutable struct Context
  cache::Union{Dict{Key,Any},Nothing}
end

Context() = Context(nothing)

cache(cx::Context) = cx.cache === nothing ? (cx.cache = Dict{Key,Any}()) : cx.cache

struct Pullback{S,T}
  t::T
end

Pullback{S}(x) where S = Pullback{S,typeof(x)}(x)

Base.show(io::IO, j::Pullback{S}) where S = print(io, "J#$(S.parameters[1])(...)")

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

# TODO the case where we return a mutable value
function _forward(f, args...)
  cx = Context()
  ks = mutkeys(f, args...)
  y, back = _forward(cx, f, args...)
  y, dy -> out_grad_mut(cx, ks, back(dy))
end

tailmemaybe(::Nothing) = nothing
tailmemaybe(x::Tuple) = Base.tail(x)

function forward(f, args...)
  y, back = _forward(f, args...)
  y, Δ -> tailmemaybe(back(Δ))
end

function gradient(f, args...)
  y, back = forward(f, args...)
  y isa Real || error("Function output is not scalar")
  return back(Int8(1))
end

derivative(f::F, x) where F = gradient(f, x)[1]

Base.adjoint(f::Function) = x -> derivative(f, x)

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

Base.getindex(gs::Grads, x) = gs.grads[Key(x)]
Base.haskey(gs::Grads, x) = haskey(gs.grads, Key(x))

function forward(f, ps::Params)
  cx = Context()
  y, back = _forward(cx, f)
  y, function (Δ)
    for p in ps
      cache(cx)[Key(p)] = nothing
    end
    back(Δ)
    Grads(cx.cache) # TODO make a copy
  end
end

# Reflection

# function code_grad(f, T)
#   forw = code_typed(_forward, Tuple{Context,Typeof(f),T.parameters...})[1]
#   Y, J = forw[2].parameters
#   back = typed_meta(Tuple{J,Y}, optimize=true)
#   back = back.code=>back.ret
#   (forw, back)
# end

# macro code_grad(ex)
#   isexpr(ex, :call) || error("@code_grad f(args...)")
#   f, args = ex.args[1], ex.args[2:end]
#   :(code_grad($(esc(f)), typesof($(esc.(args)...))))
# end
