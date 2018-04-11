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
      push!(back, Expr(:call, GlobalRef(Main, :getindex), Δ, i))
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
