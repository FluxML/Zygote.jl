using Core: CodeInfo

const usetyped = isdefined(Core.Compiler, :zygote)

worldcounter() = ccall(:jl_get_world_counter, UInt, ())

struct TypedMeta
  frame::InferenceState
  method::Method
  code::CodeInfo
  ret
end

@eval Core.Compiler function typeinf_code3(method::Method, @nospecialize(atypes), sparams::SimpleVector, run_optimizer::Bool, params::Params)
    code = code_for_method(method, atypes, sparams, params.world)
    code === nothing && return (nothing, Any)
    ccall(:jl_typeinf_begin, Cvoid, ())
    result = InferenceResult(code)
    frame = InferenceState(result, false, params)
    frame === nothing && return (nothing, Any)
    if typeinf(frame) && run_optimizer
        opt = OptimizationState(frame)
        optimize(opt, result.result)
        opt.src.inferred = true
    end
    ccall(:jl_typeinf_end, Cvoid, ())
    frame.inferred || return (nothing, Any)
    return frame
end

function typed_meta(T; world = worldcounter(), optimize = false)
  F = T.parameters[1]
  F isa DataType && (F.name.module === Core.Compiler ||
                     F <: Core.Builtin ||
                     F <: Core.Builtin) && return nothing
  _methods = Base._methods_by_ftype(T, -1, world)
  length(_methods) == 1 || return nothing
  type_signature, sps, method = first(_methods)
  params = Core.Compiler.Params(world)
  frame = Core.Compiler.typeinf_code3(method, type_signature, sps, optimize, params)
  ci = frame.src
  ci.inferred = true
  ci.method_for_inference_limit_heuristics = method
  if ci.ssavaluetypes == 0 # constant return; IRCode doesn't like this
    ci.ssavaluetypes = Any[Any]
  end
  return TypedMeta(frame, method, ci, widenconst(frame.result.result))
end

struct Meta
  method::Method
  code::CodeInfo
  sparams
end

function untyped_meta(T; world = worldcounter())
  F = T.parameters[1]
  F isa DataType && (F.name.module === Core.Compiler ||
                     F <: Core.Builtin ||
                     F <: Core.Builtin) && return nothing
  _methods = Base._methods_by_ftype(T, -1, world)
  length(_methods) == 1 || return nothing
  type_signature, sps, method = first(_methods)
  mi = Core.Compiler.code_for_method(method, type_signature, sps, world, false)
  ci = Base.isgenerated(mi) ? Compiler.get_staged(mi) : Base.uncompressed_ast(mi)
  ci.method_for_inference_limit_heuristics = method
  Meta(method, ci, sps)
end

meta(T; world = worldcounter()) =
  usetyped ?
      typed_meta(T, world = world) :
    untyped_meta(T, world = world)

function inline_sparams!(ir::IRCode, sps)
  ir = IncrementalCompact(ir)
  for (i, x) in ir
    isexpr(x, :static_parameter) && (ir[i] = sps[x.args[1]]; continue)
    u = userefs(x)
    for x in u
      isexpr(x[], :static_parameter) && (x[] = sps[x[].args[1]])
    end
    ir[i] = u[]
  end
  return finish(ir)
end

function IRCode(meta::TypedMeta)
  opt = OptimizationState(meta.frame)
  ir = just_construct_ssa(meta.code, deepcopy(meta.code.code),
                          Int(meta.method.nargs)-1, opt)
  return inline_sparams!(ir, opt.sp)
end

function IRCode(meta::Meta)
  ir = just_construct_ssa(meta.code, deepcopy(meta.code.code),
                          Int(meta.method.nargs)-1, meta.sparams)
  return inline_sparams!(ir, meta.sparams)
end

function code_ir(f, T)
  m = meta(Tuple{Typeof(f),T.parameters...})
  return IRCode(m)
end

function code_irm(ex)
  isexpr(ex, :call) || error("@code_ir f(args...)")
  f, args = ex.args[1], ex.args[2:end]
  :(code_ir($(esc(f)), typesof($(esc.(args)...))))
end

macro code_ir(ex)
  code_irm(ex)
end

function argnames!(meta, names...)
  meta.code.slotnames = [names...]
end

function spliceargs!(meta, ir::IRCode, args...)
  for i = 1:length(ir.stmts)
    ir[SSAValue(i)] = argmap(x -> Argument(x.n+length(args)), ir[SSAValue(i)])
  end
  for (name, T) in reverse(args)
    pushfirst!(ir.argtypes, T)
    pushfirst!(meta.code.slotnames, name)
  end
  return ir
end

# Behave as if the function signature is f(args...)
function varargs!(meta, ir::IRCode, n = 1)
  isva = meta.method.isva
  Ts = widenconst.(ir.argtypes[n+1:end])
  argtypes = !isva ?
    Any[ir.argtypes[1:n]..., Tuple{Ts...}] :
    Any[ir.argtypes[1:n]..., Tuple{Ts[1:end-1]...,Ts[end].parameters...}]
  empty!(ir.argtypes); append!(ir.argtypes, argtypes)
  ir = IncrementalCompact(ir)
  map = Dict{Argument,Any}()
  for i = 1:(length(Ts)-isva)
    map[Argument(i+n)] = insert_node_here!(ir, xcall(Base, :getfield, Argument(n+1), i), Ts[i], Int32(0))
  end
  if isva
    i = length(Ts)
    xs, T = Argument(n+1), argtypes[end]
    for _ = 1:i-1
      T = Tuple{T.parameters[2:end]...}
      xs = insert_node_here!(ir, xcall(Base, :tail, xs), T, Int32(0))
    end
    map[Argument(i+n)] = xs
  end
  for (i, x) in ir
    ir[i] = argmap(a -> get(map, a, a), x)
  end
  return finish_dc(ir)
end

function update!(meta, ir::IRCode)
  Core.Compiler.replace_code_newstyle!(meta.code, ir, length(ir.argtypes)-1)
  usetyped || (meta.code.ssavaluetypes = length(meta.code.code))
  slots!(meta.code)
end

@generated function roundtrip(f, args...)
  m = meta(Tuple{f,args...})
  ir = IRCode(m)
  ir = varargs!(m, ir)
  argnames!(m, :f, :args)
  ir = spliceargs!(m, ir, (Symbol("#self#"), typeof(roundtrip)))
  usetyped || (ir = slots!(ir))
  update!(m, ir)
  return m.code
end

function inlineable!(ir)
  insert_node!(ir, 1, Any, Expr(:meta, :inline))
  compact!(ir)
end

function log!(ir, msg)
  insert_node!(ir, 1, Any, xcall(Core, :println, msg))
  compact!(ir)
end
