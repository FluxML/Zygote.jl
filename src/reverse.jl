struct Forward
  id::Int
end

Base.show(io::IO, x::Forward) = print(io, "@", x.id)

# Only the final BB can return (so we have a single entry point in reverse).
# TODO: merge return nodes
validcfg(ir) =
  isexpr(ir.stmts[end], :return) &&
  !any(x -> isexpr(x, :return), ir.stmts[1:end-1])

# Insert Phi nodes which record branches taken
function record_branches!(ir::IRCode)
  for b in blocks(ir)
    ps = BasicBlock(b).preds
    length(ps) > 1 || continue
    @assert length(ps) == 2
    insert!(b, 1, Phi(ps, [false, true]))
  end
  return ir
end

function reverse_blocks(ir::IRCode)
  @assert validcfg(ir)
  rev = IRCode([], CFG([], []))
  newidx(i) = length(ir.cfg.blocks)-i+1
  for b in reverse(blocks(ir))
    succs, preds = newidx.(BasicBlock(b).preds), newidx.(BasicBlock(b).succs)
    newblock!(rev, succs = succs, preds = preds)
    @assert length(succs) ≤ 2
    if length(succs) == 1
      push!(rev, GotoNode(succs[1]))
    elseif length(succs) == 2
      push!(rev, Expr(:gotoifnot, Forward(range(b)[1]), succs[1]))
      push!(rev, GotoNode(succs[2]))
    else
      push!(rev, nothing)
    end
  end
  return rev
end
