struct Alpha
  id::Int
end

Base.show(io::IO, x::Alpha) = print(io, "@", x.id)

alpha(x) = x
alpha(x::SSAValue) = Alpha(x.id)
alpha(x::Argument) = Argument(x.n+1)

struct Delta
  id::Int
end

Base.show(io::IO, x::Delta) = print(io, "Δ", x.id)

# Only the final BB can return (so we have a single entry point in reverse).
# TODO: merge return nodes
validcfg(ir) =
  ir.stmts[end] isa ReturnNode &&
  !any(x -> x isa ReturnNode, ir.stmts[1:end-1])

# # Insert Phi nodes which record branches taken
function record_branches(ir::IRCode)
  ir = IncrementalCompact(ir)
  for (i, x) in ir
    bi = findfirst(x -> x == i+1, ir.ir.cfg.index)
    bi == nothing && continue
    preds = ir.ir.cfg.blocks[bi+1].preds
    length(preds) > 1 || continue
    @assert length(preds) <= 2
    insert_node_here!(ir, PhiNode(preds, [false, true]), Bool, ir.result_lines[i])
  end
  return finish(ir)
end

newbi(ir, b) = length(ir.cfg.blocks)-b+1

function reverse_blocks(ir::IRCode)
  stmts = []
  blocks = []
  for (i, b) in enumerate(reverse(ir.cfg.blocks))
    preds = newbi.(ir, b.succs)
    succs = newbi.(ir, b.preds)
    st = length(stmts)+1
    if isempty(succs)
      push!(stmts, nothing)
    elseif length(succs) == 1
      push!(stmts, GotoNode(succs[1]))
    else
      push!(stmts, GotoIfNot(Alpha(1), succs[1]))
      push!(stmts, GotoNode(succs[2]))
    end
    push!(blocks, BasicBlock(StmtRange(st,length(stmts)), preds, succs))
  end
  return IRCode(ir, stmts, Any[Any for _ in stmts], [-1 for _ in stmts], CFG(blocks), NI.NewNode[])
end
