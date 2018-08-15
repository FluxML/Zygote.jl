struct Slot
  id::Symbol
end

function insert_blockend!(ir::IRCode, pos, typ, val)
  i = last(ir.cfg.blocks[pos].stmts)
  while ir.stmts[i] isa Union{GotoNode,GotoIfNot}
    i -= 1
  end
  insert_node!(ir, i, typ, val, true)
end

function slots!(ir::IRCode)
  n = 0
  for b = 1:length(ir.cfg.blocks)
    i = first(ir.cfg.blocks[b].stmts)
    while (phi = ir[SSAValue(i)]) isa PhiNode
      slot = Slot(Symbol(:phi, n+=1))
      ir[SSAValue(i)] = slot
      for (pred, val) in zip(phi.edges, phi.values)
        insert_blockend!(ir, pred, Any, :($slot = $val))
      end
      i += 1
    end
  end
  return compact!(ir)
end

using Core.Compiler: CodeInfo, SlotNumber

function slots!(ci::CodeInfo)
  ss = Dict{Slot,SlotNumber}()
  for i = 1:length(ci.code)
    ci.code[i] = MacroTools.prewalk(ci.code[i]) do x
      x isa Slot || return x
      haskey(ss, x) && return ss[x]
      push!(ci.slotnames, x.id)
      push!(ci.slotflags, 0x00)
      ss[x] = SlotNumber(length(ci.slotnames))
    end
  end
  return ci
end
