struct Context
  grads::IdDict{Any,Any}
end

Context() = Context(IdDict())

struct J{S,T}
  t::T
end

J{S}(x) where S = J{S,typeof(x)}(x)

Base.show(io::IO, j::J{S}) where S = print(io, "J{$(S.parameters[1])}(...)")

# interface2.jl

# Wrappers

_forward(f, args...) = _forward(Context(), f, args...)

tailmemaybe(::Nothing) = nothing
tailmemaybe(x::Tuple) = Base.tail(x)

function forward(f, args...)
  y, back = _forward(f, args...)
  y, Δ -> tailmemaybe(back(Δ))
end

function gradient(f, args...)
  y, J = forward(f, args...)
  y isa Real || error("Function output is not scalar")
  return J(1)
end

# Reflection

function code_grad(f, T)
  forw = code_typed(_forward, Tuple{Context,typeof(f),T.parameters...})[1]
  Y, J = forw[2].parameters
  back = typed_meta(Tuple{J,Y}, optimize=true)
  back = back.code=>back.ret
  (forw, back)
end

macro code_grad(ex)
  isexpr(ex, :call) || error("@code_grad f(args...)")
  f, args = ex.args[1], ex.args[2:end]
  :(code_grad($(esc(f)), typesof($(esc.(args)...))))
end
