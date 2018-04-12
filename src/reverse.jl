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

function grad_ex!(back, forw, i, grads)
  ex = forw.stmts[i]
  grad(x, Δ) = push!(get!(grads, x, []), Δ)
  grad(x) = grad(x, SSAValue(length(back.stmts)))
  if ex isa GotoNode || isexpr(ex, :gotoifnot) || ex == nothing
  elseif isexpr(ex, :return)
    grad(ex.args[1], Argument(2))
  elseif ex isa Phi
    push!(back, Delta(i))
    grad.(ex.values)
  elseif isexpr(ex, :call)
    args = ex.args[2:end]
    push!(back, Expr(:call, :∇, ex.args[1], Delta(i), alpha.(args)...))
    Δ = SSAValue(length(back.stmts))
    for (i, x) in enumerate(args)
      push!(back, Expr(:call, GlobalRef(Base, :getindex), Δ, i))
      grad(x)
    end
  else
    error("Can't handle $ex")
  end
end

function reverse_blocks(forw::IRCode)
  @assert validcfg(forw)
  grads = Dict()
  back = IRCode([], CFG([], []))
  newidx(i) = length(forw.cfg.blocks)-i+1
  for b in reverse(blocks(forw))
    succs, preds = newidx.(BasicBlock(b).preds), newidx.(BasicBlock(b).succs)
    newblock!(back, succs = succs, preds = preds)
    for i in reverse(range(b))
      grad_ex!(back, forw, i, grads)
    end
    @assert length(succs) ≤ 2
    if length(succs) == 1
      push!(back, GotoNode(succs[1]))
    elseif length(succs) == 2
      push!(back, Expr(:gotoifnot, Alpha(range(b)[1]), succs[1]))
      push!(back, GotoNode(succs[2]))
    end
    isempty(blocks(back)[end]) && push!(back, nothing)
  end
  return back, grads
end

# TODO: need a much better reaching check here.
# This assumes that all variable uses will occur in either the same BB
# or an immediate successor.
function reachinggrads(ir, grads, idx)
  b = blockat(ir, idx)
  gs = []
  ps = [[] for p in preds(b)]
  for grad in grads
    if grad isa Argument || grad.id in range(b)[1]:idx
      push!(gs, grad)
    elseif all(p -> grad.id in range(p), preds(b))
      push!(gs, grad)
    elseif !any(p -> grad.id in range(p), preds(b))
      error("$grad not found")
    else
      for (i, p) in enumerate(preds(b))
        grad.id in range(p) && push!(ps[i], grad)
      end
    end
  end
  return gs, ps
end

# TODO: we only need `grads` here to keep SSAValue numbers in sync,
# which seems pretty unclean.
function fillphis!(ir, grads, ps, i)
  # while any(!isempty, ps)
  #   error("need phis")
  # end
  return []
end

function fill_deltas!(ir, grads)
  function _fill_deltas(x, i)
    haskey(grads, x) || return x
    gs, ps = reachinggrads(ir, grads[x], i)
    gs = [gs..., fillphis!(ir, grads, ps, i-1)...]
    return reduce((a, b) -> :(accum($a, $b)), gs)
  end
  for i = 1:length(ir.stmts)
    fill_deltas(x) = x
    fill_deltas(x::Delta) = _fill_deltas(SSAValue(x.id), i)
    fill_deltas(x::Expr) = isexpr(x, :call) ? Expr(:call, fill_deltas.(x.args)...) : x
    ir.stmts[i] = fill_deltas(ir.stmts[i])
  end
  push!(ir, _fill_deltas(Argument(2), length(ir.stmts)))
  return back
end
