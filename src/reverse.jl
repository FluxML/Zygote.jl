# Only the final BB can return (so we have a single entry point in reverse).
# TODO: merge return nodes
validcfg(ir) =
  isexpr(ir.stmts[end], :return) &&
  !any(x -> isexpr(x, :return), ir.stmts[1:end-1])

function recbranches!(ir::IRCode)
  for b in blocks(ir)
    ps = BasicBlock(b).preds
    length(ps) > 1 || continue
    @assert length(ps) == 2
    insert!(b, 1, Phi(ps, [true, false]))
  end
  return ir
end

reverse(x) = Base.reverse(x)

function reverse(ir::IRCode)
  rev = IRCode([], CFG([], []))
  bs = revblocks(ir.cfg)
  newidx(i) = length(ir.cfg.blocks)-i+1
  for b in reverse(blocks(ir))
    newblock!(rev, succs = newidx.(BasicBlock(b).preds), preds = newidx.(BasicBlock(b).succs))
    for i in reverse(range(b))
      ex = ir.stmts[i]
      (isexpr(ex, :gotoifnot) || ex isa GotoNode) && continue
      push!(rev, ir.stmts[i])
    end
  end
  return rev
end
