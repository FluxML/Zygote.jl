using Base: @get!

iscall(x, m::Module, n::Symbol) = isexpr(x, :call) && x.args[1] == GlobalRef(m, n)

function isassert(ir, i)
  ex = ir.stmts[i+3]
  iscall(ex, Zygote, :typeassert)
end

function merge_returns(ir)
  any(x -> x == unreachable, ir.stmts) && error("`throw` not supported")
  rs = findall(x -> x isa ReturnNode, ir.stmts)
  length(rs) <= 1 && return ir
  bs = blockidx.(Ref(ir), rs)
  xs = []
  bb = length(ir.cfg.blocks)+1
  @assert length(unique(bs)) == length(bs)
  for r in rs
    x = ir.stmts[r].val
    x == GlobalRef(Base, :nothing) && (x = nothing)
    @assert !(x isa Union{GlobalRef,Expr})
    push!(xs, x)
    ir.stmts[r] = GotoNode(bb)
  end
  push!(ir.cfg.blocks, BasicBlock(StmtRange(length(ir.stmts), length(ir.stmts)-1), bs, []))
  push!(ir.cfg.index, length(ir.stmts))
  for b in bs
    push!(ir.cfg.blocks[b].succs, bb)
  end
  ir = IncrementalCompact(ir)
  for _ in ir end
  # TODO preserve types
  r = insert_node_here!(ir, PhiNode(bs, xs), Any, ir.result_lines[end])
  insert_node_here!(ir, ReturnNode(r), Any, ir.result_lines[end])
  ir = finish(ir)
  return ir
end

struct Alpha
  id::Int
end

Base.show(io::IO, x::Alpha) = print(io, "@", x.id)

alpha(x) = x
alpha(x::SSAValue) = Alpha(x.id)

gradindex(x, i) = x[i]
gradindex(::Nothing, i) = nothing
xgetindex(x, i...) = Expr(:call, GlobalRef(Base, :getindex), x, i...)
xgradindex(x, i) = xcall(Zygote, :gradindex, x, i)

function record_branches!(ir::IRCode)
  ir = IncrementalCompact(ir)
  offset = 0
  for (i, x) in ir
    bi = findfirst(x -> x == i+1-offset, ir.ir.cfg.index)
    bi == nothing && continue
    preds = ir.ir.cfg.blocks[bi+1].preds
    length(preds) > 1 || continue
    @assert length(preds) <= 2 "> 2 predecessors not implemented"
    insert_node_here!(ir, PhiNode(sort(preds), [false, true]), Bool, ir.result_lines[i])
    offset += 1
  end
  return finish_dc(ir)
end

istrackable(x) =
  x isa GlobalRef && x.mod ∉ (Base, Core) &&
  !(isconst(x.mod, x.name) && typeof(getfield(x.mod, x.name)) <: Union{Function,Type})

function record_globals!(ir::IRCode)
  for i = 1:length(ir.stmts)
    ex = ir[SSAValue(i)]
    # TODO general globalrefs
    if isexpr(ex, :call)
      for j = 1:length(ex.args)
        istrackable(ex.args[j]) || continue
        ex.args[j] = insert_node!(ir, i, Any, xcall(Zygote, :unwrap, ex.args[j]))
      end
    end
  end
  return compact!(ir)
end

ignored_f(f) = f in (GlobalRef(Base, :not_int),
                     GlobalRef(Core.Intrinsics, :not_int),
                     GlobalRef(Core, :(===)),
                     GlobalRef(Core, :apply_type),
                     GlobalRef(Core, :typeof))
ignored_f(ir, f) = ignored_f(f)
ignored_f(ir, f::SSAValue) = ignored_f(ir[f])

ignored(ir, ex) = isexpr(ex, :call) && ignored_f(ir, ex.args[1])
ignored(ir, ex::SSAValue) = ignored(ir, ir[ex])

function valid_usages(ir)
  r = Dict()
  for (x, us) in usages(ir)
    x isa Union{SSAValue,Argument} || continue
    us′ = filter(i -> !ignored(ir, i), us)
    isempty(us′) || (r[x] = us′)
  end
  return r
end

reachable(ir) = keys(valid_usages(ir))

# TODO: remove this once we don't mess with type inference
function _forward_type(Ts)
  isconcretetype(Tuple{Ts...}) || return Any
  typed_meta(Tuple{Ts...}) == nothing && return Any
  T = Core.Compiler.return_type(_forward, Tuple{Context,Ts...})
  return T == Union{} ? Any : T
end

isvalidtype(jT, yT) = jT <: Tuple && length(jT.parameters) == 2 && jT.parameters[1] <: yT

function record!(ir::IRCode)
  pushfirst!(ir.argtypes, typeof(_forward), Context)
  xs = reachable(ir)
  for i = 1:length(ir.stmts)
    ex = argmap(x -> Argument(x.n+2), ir[SSAValue(i)])
    isexpr(ex, :new) && (ex = ir[SSAValue(i)] = xcall(Zygote, :__new__, ex.args...))
    if isexpr(ex, :call) && !ignored(ir, ex)
      yT = widenconst(types(ir)[i])
      T = _forward_type(exprtype.(Ref(ir), ex.args))
      if T == Any || isvalidtype(T, yT)
        yJ = insert_node!(ir, i, T, xcall(Zygote, :_forward, Argument(2), ex.args...))
        ir[SSAValue(i)] = xgetindex(yJ, 1)
        insert_node!(ir, i, T == Any ? Any : T.parameters[2], xgetindex(yJ, 2), true)
      else
        yJ = insert_node!(ir, i, Any, xcall(Zygote, :_forward, Argument(2), ex.args...))
        y =  insert_node!(ir, i, Any, xgetindex(yJ, 1))
        J =  insert_node!(ir, i, Any, xgetindex(yJ, 2))
        ir[SSAValue(i)] = xcall(Zygote, :typeassert, y, yT)
      end
    else
      ir[SSAValue(i)] = ex
    end
  end
  ir, m = _compact!(ir)
  return ir, Set(x isa Argument ? Argument(x.n+2) : x for x in rename(xs, m))
end

# Backwards Pass

function reverse_cfg(cfg, perm)
  newidx(i) = invperm(perm)[i]
  CFG([BasicBlock(StmtRange(1,0),newidx.(b.succs),newidx.(b.preds)) for b in cfg.blocks[perm]])
end

function reverse_order(cfg)
  n = length(cfg.blocks)
  perm = n:-1:1
  guess = reverse_cfg(cfg, perm)
  dt = construct_domtree(guess)
  perm[sortperm(1:n, by = x -> dt.nodes[x].level)]
end

struct Primal
  forw::IRCode
  perm::Vector{Int}
  wrt::Set{Any}
  varargs::Union{Int,Nothing}
end

Primal(ir::IRCode, xs, vs) = Primal(ir, reverse_order(ir.cfg), xs, vs)

function Primal(ir::IRCode; varargs = nothing)
  ir = merge_returns(ir)
  forw, xs = record!(record_branches!(record_globals!(ir)))
  Primal(forw, xs, varargs)
end

function blockinfo(pr::Primal)
  info = Dict(b => (phis=Dict(),partials=[],grads=[]) for b in 1:length(pr.forw.cfg.blocks))
  for b in 1:length(pr.forw.cfg.blocks), i in pr.forw.cfg.blocks[b].stmts
    ex = pr.forw[SSAValue(i)]
    if ex isa PhiNode
      for (c, x) in zip(ex.edges, ex.values)
        x in pr.wrt && push!(@get!(info[b].phis, c, []), x)
      end
    elseif iscall(ex, Zygote, :_forward)
      push!(info[b].grads, SSAValue(i+1))
      for x in ex.args[3:end]
        x in pr.wrt && push!(info[b].partials, x)
      end
    end
  end
  worklist = collect(1:length(pr.forw.cfg.blocks))
  while !isempty(worklist)
    b = pop!(worklist)
    for c in pr.forw.cfg.blocks[b].preds
      in = union(get(info[b].phis, c, []), setdiff(info[b].partials, info[b].grads))
      out = union(info[c].partials, info[c].grads)
      @assert isempty(setdiff(in, out))
    end
  end
  return info
end

function IRCode(ir::Primal)
  stmts = []
  blocks = []
  newidx(i) = invperm(ir.perm)[i]
  for block in ir.perm
    old = ir.forw.cfg.blocks[block]
    start = length(stmts)+1
    block == length(ir.perm) && push!(stmts, :(Δ()))
    preds, succs = newidx.(old.succs), newidx.(sort(old.preds))
    if isempty(succs)
      push!(stmts, nothing)
    elseif length(succs) == 1
      push!(stmts, GotoNode(succs[1]))
    else
      push!(stmts, GotoIfNot(Alpha(range(old)[1]), succs[1]))
      block+1 != succs[2] && push!(stmts, GotoNode(succs[2]))
    end
    push!(blocks, BasicBlock(StmtRange(start,length(stmts)), preds, succs))
  end
  ir = IRCode(ir.forw, stmts, Any[Any for _ in stmts], Int32[0 for _ in stmts],
              [0x00 for _ in stmts], CFG(blocks), NewNode[])
end

function reverse_ir(pr::Primal)
  ir = IRCode(pr)
  grads = Dict()
  partials = Dict(x => [] for x in pr.wrt)
  for b in pr.perm, i in reverse(pr.forw.cfg.blocks[b].stmts)
    j = ir.cfg.blocks[invperm(pr.perm)[b]].stmts[1]
    j = max(j, 2)
    ex = pr.forw[SSAValue(i)]
    if ex isa ReturnNode
      push!(partials[ex.val], SSAValue(1))
    elseif ex isa PhiNode
      any(x -> x in pr.wrt, ex.values) || continue
      Δ = insert_node!(ir, j, Any, xcall(Zygote, :accum))
      grads[SSAValue(i)] = Δ
      for x in ex.values
        x in pr.wrt || continue
        push!(partials[x], Δ)
      end
    elseif iscall(ex, Zygote, :_forward)
      # TODO remove with type hacks above
      y = isassert(pr.forw, i) ? SSAValue(i+3) : SSAValue(i+1)
      y in pr.wrt || continue
      J = Alpha(i+2)
      Δ = insert_node!(ir, j, Any, xcall(Zygote, :accum))
      insert_node!(ir, j, Any, Expr(:call, J, Δ))
      grads[y] = Δ
      for (i, x) in enumerate(ex.args[3:end])
        x in pr.wrt || continue
        dx = insert_node!(ir, j, Any, xgradindex(Δ, i))
        push!(partials[x], dx)
      end
    end
  end
  ir, m = _compact!(ir)
  return ir, rename(grads, m), rename(partials, m)
end

function accumulators!(pr::Primal, ir::IRCode, grads, partials)
  blockpartials(b, x) = filter(x -> x.id in ir.cfg.blocks[b].stmts, partials[x])
  accums = Dict()
  info = blockinfo(pr)
  for b = 1:length(ir.cfg.blocks), x in setdiff(info[b].partials, info[b].grads)
    ps = blockpartials(pr.perm[b], x)
    p = insert_blockend!(ir, pr.perm[b], Any, xcall(Zygote, :accum, ps...))
    setdiff!(partials[x], ps)
    push!(partials[x], p)
    accums[(pr.perm[b],x)] = p
  end
  blockpartial(b, x) = haskey(accums, (b, x)) ? accums[(b, x)] : get(blockpartials(b, x), 1, nothing)
  for ((b, x), p) in accums
    preds = ir.cfg.blocks[b].preds
    ps = map(b -> blockpartial(b, x), preds)
    if length(preds) > 1
      p = insert_blockstart!(ir, b, Any, PhiNode(preds, ps))
    else
      p = ps[1]
    end
  end
  for (x, dx) in grads
    b = blockidx(ir, dx)
    append!(ir[dx].args, blockpartials(b, x))
    preds = ir.cfg.blocks[b].preds
    length(preds) > 0 || continue
    ps = map(b -> blockpartial(b, x), preds)
    any(x -> x != nothing, ps) || continue
    if length(preds) > 1
      p = insert_blockstart!(ir, b, Any, PhiNode(preds, ps))
    else
      p = ps[1]
    end
    push!(ir[dx].args, blockpartials(b, x)..., p)
  end
  return compact!(ir)
end

# let
#   pr = Primal(@code_ir myabs(2))
#   accumulators!(pr, reverse_ir(pr)...)
# end

@inline tuple_va(N, xs) = xs
@inline tuple_va(N, x, xs...) = (x, tuple_va(N, xs...)...)
@inline tuple_va(::Val{N}, ::Nothing) where N = ntuple(_ -> nothing, Val(N))

# struct Adjoint
#   forw::IRCode
#   back::IRCode
#   perm::Vector{Int}
# end

using InteractiveUtils: @which

macro adjoint(ex)
  :(grad_ir($(code_irm(ex)), varargs = varargs($(esc(:(@which $ex))), length(($(esc.(ex.args)...),)))))
end

function pow(x, n)
  r = 1
  while n > 0
    n -= 1
    r *= x
  end
  return r
end

function myabs(x)
  if x < 0
    x = -x
  end
  return x
end
