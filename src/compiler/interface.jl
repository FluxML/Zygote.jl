mutable struct Context
  cache::Union{IdDict{Any,Any},Nothing}
end

Context() = Context(nothing)

cache(cx::Context) = cx.cache == nothing ? (cx.cache = IdDict()) : cx.cache

struct J{S,T}
  t::T
end

J{S}(x) where S = J{S,typeof(x)}(x)

Base.show(io::IO, j::J{S}) where S = print(io, "J($(j.t[1]))")

# interface2.jl

# Wrappers

_forward(f, args...) = _forward(Context(), f, args...)

tailmemaybe(::Nothing) = nothing
tailmemaybe(x::Tuple) = Base.tail(x)

function forward(f, args...)
  y, back = _forward(f, args...)
  y, Δ -> tailmemaybe(back(Δ))
end

isscalar(y) = y isa Real
function gradient(f, args...)
  y, J = forward(f, args...)
  isscalar(y) || error("Function output is not scalar")
  return J(1)
end

derivative(f, x) = gradient(f, x)[1]

Base.adjoint(f::Function) = x -> derivative(f, x)

# Param-style wrappers

# TODO store ids only
struct Params
  params::IdSet{Any}
  Params(xs) = new(IdSet(xs))
end

@forward Params.params Base.iterate

struct Grads
  grads::IdDict{Any,Any}
end

Base.show(io::IO, ps::Grads) = print(io, "Grads(...)")

@forward Grads.grads Base.setindex!, Base.getindex, Base.haskey

function forward(f, ps::Params)
  cx = Context()
  y, back = _forward(cx, f)
  y, function (Δ)
    for p in ps
      cache(cx)[p] = nothing
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
