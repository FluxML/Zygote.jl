struct Context
end

function lookup(T, world = ccall(:jl_get_world_counter, UInt, ()); optimize = false)
  F = T.parameters[1]
  (F.name.module === Core.Compiler || F <: Core.Builtin || F <: Core.Builtin) && return nothing
  _methods = Base._methods_by_ftype(T, -1, world)
  length(_methods) == 1 || return nothing
  type_signature, static_params, method = first(_methods)
  params = Core.Compiler.Params(world)
  (_, ci, ty) = Core.Compiler.typeinf_code(method, type_signature, static_params, optimize, optimize, params)
  ir = compact!(just_construct_ssa(ci, copy(ci.code), Int(method.nargs)-1, [NullLineInfo]))
  ir, collect(static_params), method.nargs, method.isva
end

function _lookup_grad(T)
  (func = lookup(T)) == nothing && return
  ir, sparams, nargs, isva = func
  forw, back = stacks!(grad_ir(ir, varargs = isva))
  forw, back, isva, nargs, sparams
end

struct J{S,T}
  t::T
end

J{S}(x) where S = J{S,typeof(x)}(x)

Base.show(io::IO, j::J{S}) where S = print(io, "J{$(S.parameters[1])}(...)")

function _forward(ctx::Context, f, args...)
  T = typesof(f, args...)
  (g = _lookup_grad(T)) == nothing &&
    return f(args...), Δ -> error("Undifferentiable function $f")
  forw, _, isva, nargs, sparams = g
  isva && (args = (args[1:nargs-2]...,args[nargs-1:end]))
  y, c = interpret(forw, _forward, ctx, f, args..., sparams = sparams)
  return y, J{T}(c)
end

_forward(args...) = _forward(Context(), args...)

function (j::J{T})(Δ) where T
  (g = _lookup_grad(T)) == nothing && return map(_ -> nothing, j.t)
  _, back, = g
  return interpret(back, j.t, Δ)
end

function forward(f, args...)
  y, back = _forward(f, args...)
  y, function (Δ)
    Δ = back(Δ)
    Δ == nothing && return
    Base.tail(Δ)
  end
end

function gradient(f, args...)
  y, J = forward(f, args...)
  y isa Real || error("Function output is not scalar")
  return J(1)
end
