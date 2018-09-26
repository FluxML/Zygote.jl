using Base: @get!

@inline tuple_va(N, xs) = xs
@inline tuple_va(N, x, xs...) = (x, tuple_va(N, xs...)...)
@inline tuple_va(::Val{N}, ::Nothing) where N = ntuple(_ -> nothing, Val(N))

iscall(x, m::Module, n::Symbol) = isexpr(x, :call) && x.args[1] == GlobalRef(m, n)

function isassert(ir, i)
  ex = ir.stmts[i+3]
  iscall(ex, Zygote, :typeassert)
end

# TODO: Move this to Base
function append_node!(ir, @nospecialize(typ), @nospecialize(node), line)
  push!(ir.stmts, node)
  push!(ir.types, typ)
  push!(ir.lines, line)
  push!(ir.flags, 0)
  return SSAValue(length(ir.stmts))
end

function merge_returns(ir)
  any(x -> x == unreachable, ir.stmts) && error("`throw` not supported")
  rs = findall(x -> x isa ReturnNode && isdefined(x, :val), ir.stmts)
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
  for b in bs
    push!(ir.cfg.blocks[b].succs, bb)
  end
  push!(ir.cfg.blocks, BasicBlock(StmtRange(length(ir.stmts)+1, length(ir.stmts)+3), bs, []))
  push!(ir.cfg.index, length(ir.stmts) + 1)
  r = append_node!(ir, Any, PhiNode(bs, xs), ir.lines[end])
  append_node!(ir, Any, ReturnNode(r), ir.lines[end])
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
    @assert length(preds) <= 2 "not implemented"
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
  @Base.show ir
  return compact!(ir)
end

ignored_f(f) = f in (GlobalRef(Base, :not_int),
                     GlobalRef(Core.Intrinsics, :not_int),
                     GlobalRef(Core, :(===)),
                     GlobalRef(Core, :apply_type),
                     GlobalRef(Core, :typeof),
                     GlobalRef(Core, :throw),
                     GlobalRef(Base, :kwerr),
                     GlobalRef(Core, :kwfunc))
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

# Hack to work around fragile constant prop through overloaded functions
is_literal_getproperty(ex) = iscall(ex, Base, :getproperty) && ex.args[3] isa QuoteNode

# TODO: remove this once we don't mess with type inference
function _forward_type(Ts)
  usetyped || return Any
  all(T -> isconcretetype(T) || T <: DataType, Ts) || return Any
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
    is_literal_getproperty(ex) &&
      (ex = ir[SSAValue(i)] = xcall(Zygote, :literal_getproperty, ex.args[2], Val(ex.args[3].value)))
    if isexpr(ex, :call) && !ignored(ir, ex)
      yT = widenconst(types(ir)[i])
      T = _forward_type(exprtype.(Ref(ir), ex.args))
      if yT == Any || isvalidtype(T, yT)
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

newblock(pr::Primal, b) = invperm(pr.perm)[b]
oldblock(pr::Primal, b) = pr.perm[b]

function blockinfo(pr::Primal)
  preds(b) = pr.forw.cfg.blocks[b].preds
  info = Dict(b => (phis=Dict(),partials=[],grads=[]) for b in 1:length(pr.forw.cfg.blocks))
  append!(info[1].grads, filter(x -> x isa Argument, pr.wrt))
  for b in 1:length(pr.forw.cfg.blocks), i in pr.forw.cfg.blocks[b].stmts
    ex = pr.forw[SSAValue(i)]
    if ex isa ReturnNode
      ex.val in pr.wrt && push!(info[b].partials, ex.val)
    elseif ex isa PiNode
      (SSAValue(i) in pr.wrt && ex.val in pr.wrt) || continue
      push!(info[b].grads, SSAValue(i))
      push!(info[b].partials, ex.val)
    elseif ex isa PhiNode
      any(x -> x in pr.wrt, ex.values) && push!(info[b].grads, SSAValue(i))
      for (c, x) in zip(ex.edges, ex.values)
        x in pr.wrt && push!(@get!(info[b].phis, c, []), x)
      end
    elseif iscall(ex, Zygote, :_forward)
      y = isassert(pr.forw, i) ? SSAValue(i+3) : SSAValue(i+1)
      push!(info[b].grads, y)
      for x in ex.args[3:end]
        x in pr.wrt && push!(info[b].partials, x)
      end
    end
  end
  worklist = collect(1:length(pr.forw.cfg.blocks))
  while !isempty(worklist)
    b = pop!(worklist)
    for c in preds(b)
      in = union(get(info[b].phis, c, []), setdiff(info[b].partials, info[b].grads))
      out = union(info[c].partials, info[c].grads)
      new = setdiff(in, out)
      if !isempty(new)
        append!(info[c].partials, new)
        c ∉ worklist && push!(worklist, c)
      end
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
    @assert length(succs) <= 2
    if isempty(succs)
      push!(stmts, nothing)
    elseif length(succs) == 1
      push!(stmts, GotoNode(succs[1]))
    else
      push!(stmts, GotoIfNot(Alpha(range(old)[1]), succs[1]))
      push!(stmts, GotoNode(succs[2]))
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
  phis = Dict()
  for b in pr.perm
    j = ir.cfg.blocks[newblock(pr, b)].stmts[1]
    j = max(j, 2)
    for i in reverse(pr.forw.cfg.blocks[b].stmts)
      ex = pr.forw[SSAValue(i)]
      if ex isa ReturnNode
        ex.val in pr.wrt && push!(partials[ex.val], SSAValue(1))
      elseif ex isa PiNode
        (SSAValue(i) in pr.wrt && ex.val in pr.wrt) || continue
        Δ = insert_node!(ir, j, Any, xcall(Zygote, :accum))
        ir.lines[j] = pr.forw.lines[i]
        grads[SSAValue(i)] = Δ
        push!(partials[ex.val], Δ)
      elseif ex isa PhiNode
        any(x -> x in pr.wrt, ex.values) || continue
        Δ = insert_node!(ir, j, Any, xcall(Zygote, :accum))
        ir.lines[j] = pr.forw.lines[i]
        grads[SSAValue(i)] = Δ
        for (c, x) in zip(ex.edges, ex.values)
          x in pr.wrt || continue
          @assert !haskey(phis, (newblock(pr, b), newblock(pr, c), x)) "not implemented"
          phis[(newblock(pr, b), newblock(pr, c), x)] = Δ
        end
      elseif iscall(ex, Zygote, :_forward)
        # TODO remove with type hacks above
        y = isassert(pr.forw, i) ? SSAValue(i+3) : SSAValue(i+1)
        J = Alpha(i+2)
        dy = insert_node!(ir, j, Any, xcall(Zygote, :accum))
        ir.lines[j] = pr.forw.lines[i]
        dxs = insert_node!(ir, j, Any, Expr(:call, J, dy))
        ir.lines[j] = pr.forw.lines[i]
        grads[y] = dy
        for (i, x) in enumerate(ex.args[3:end])
          x in pr.wrt || continue
          dx = insert_node!(ir, j, Any, xgradindex(dxs, i))
          ir.lines[j] = pr.forw.lines[i]
          push!(partials[x], dx)
        end
      end
    end
    if b == 1
      gs = []
      for i = 3:length(pr.forw.argtypes)
        Argument(i) in pr.wrt || (push!(gs, nothing); continue)
        dx = insert_node!(ir, j, Any, xcall(Zygote, :accum))
        grads[Argument(i)] = dx
        push!(gs, dx)
      end
      if pr.varargs == nothing
        Δ = insert_node!(ir, j, Any, xcall(Zygote, :tuple, gs...))
      else
        Δ = insert_node!(ir, j, Any, xcall(Zygote, :tuple_va, Val(pr.varargs), gs...))
      end
      insert_node!(ir, j, Any, ReturnNode(Δ))
    end
  end
  ir, m = _compact!(ir)
  return ir, rename(grads, m), rename(partials, m), rename(phis, m)
end

function simplify!(ir)
  ir = IncrementalCompact(ir)
  for (i, x) in ir
    iscall(x, Zygote, :accum) || continue
    filter!(x -> x != nothing, x.args)
    nargs = length(x.args)-1
    ir[i] = nargs == 0 ? nothing : nargs == 1 ? x.args[end] : x
  end
  return finish(ir)
end

function accumulators!(pr::Primal, ir::IRCode, grads, partials, phis)
  blockpartials(b, x) = filter(x -> x.id in ir.cfg.blocks[b].stmts, get(partials, x, []))
  accums = Dict()
  info = blockinfo(pr)
  for b = 1:length(ir.cfg.blocks), x in setdiff(info[b].partials, info[b].grads)
    ps = blockpartials(newblock(pr, b), x)
    p = insert_blockend!(ir, newblock(pr, b), Any, xcall(Zygote, :accum, ps...))
    setdiff!(partials[x], ps)
    push!(partials[x], p)
    accums[(newblock(pr, b),x)] = p
  end

  function predpartial(b, x)
    function blockpartial(b, c, x)
      if haskey(accums, (b, x))
        @assert !haskey(phis, (b, c, x)) "not implemented"
        return accums[(b, x)]
      elseif haskey(phis, (b, c, x))
        return phis[b, c, x]
      end
    end
    preds = ir.cfg.blocks[b].preds
    isempty(preds) && return
    ps = map(c -> blockpartial(c, b, x), preds)
    all(==(nothing), ps) && return
    length(ps) == 1 ? ps[1] : insert_blockstart!(ir, b, Any, PhiNode(preds, ps))
  end

  for ((b, x), p) in accums
    push!(ir[p].args, predpartial(b, x))
  end
  for (x, dx) in grads
    b = blockidx(ir, dx)
    append!(ir[dx].args, blockpartials(b, x))
    push!(ir[dx].args, predpartial(b, x))
  end
  return simplify!(ir)
end

struct Adjoint
  forw::IRCode
  back::IRCode
  perm::Vector{Int}
end

function Adjoint(pr::Primal)
  back = accumulators!(pr, reverse_ir(pr)...)
  Adjoint(pr.forw, compact!(compact!(back)), pr.perm)
end

Adjoint(ir::IRCode; varargs = nothing) = Adjoint(Primal(ir, varargs = varargs))

using InteractiveUtils: @which

macro adjoint(ex)
  :(Adjoint($(code_irm(ex)), varargs = varargs($(esc(:(@which $ex))), length(($(esc.(ex.args)...),)))))
end
