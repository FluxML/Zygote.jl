meta(T) = (usetyped ? IRTools.typed_meta : IRTools.meta)(T)

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
  meta.code.slotflags = [0x00 for name in names]
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

function pis!(ir::IRCode)
  for i = 1:length(ir.stmts)
    ex = ir.stmts[i]
    ex isa PiNode || continue
    ir.stmts[i] = xcall(Core, :typeassert, ex.val, ex.typ)
  end
  return ir
end

function slots!(ir::IRCode)
  n = 0
  for b = 1:length(ir.cfg.blocks)
    i = first(ir.cfg.blocks[b].stmts)
    while (phi = ir[SSAValue(i)]) isa PhiNode
      slot = IRTools.Slot(Symbol(:phi, n+=1))
      ir[SSAValue(i)] = slot
      for (pred, val) in zip(phi.edges, phi.values)
        insert_blockend!(ir, pred, Any, :($slot = $val))
      end
      i += 1
    end
  end
  return compact!(ir)
end

@generated function roundtrip(f, args...)
  m = meta(Tuple{f,args...})
  ir = IRCode(m)
  ir = varargs!(m, ir)
  argnames!(m, :f, :args)
  ir = spliceargs!(m, ir, (Symbol("#self#"), typeof(roundtrip)))
  ir = slots!(pis!(ir))
  return IRTools.update!(m, ir)
end

function inlineable!(ir)
  insert_node!(ir, 1, Any, Expr(:meta, :inline))
  compact!(ir)
end

function log!(ir, msg)
  insert_node!(ir, 1, Any, xcall(Core, :println, msg))
  compact!(ir)
end
