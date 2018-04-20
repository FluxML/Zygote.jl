using NotInferenceDontLookHere
import NotInferenceDontLookHere: IRCode, CFG, BasicBlock, Argument, ReturnNode,
  GotoIfNot, PhiNode, StmtRange, IncrementalCompact, insert_node!, insert_node_here!,
  compact!, finish, DomTree, construct_domtree, dominates, userefs
using InteractiveUtils: typesof
using Core: SSAValue, GotoNode

for T in [:IRCode, :IncrementalCompact]
  @eval begin
    Base.getindex(ir::$T, x) = NI.getindex(ir, x)
    Base.setindex!(ir::$T, x, i) = NI.setindex!(ir, x, i)
  end
end
Base.getindex(u::NI.UseRef) = NI.getindex(u)
Base.getindex(r::StmtRange, i) = (r.first:r.last)[i]

for T in :[UseRefIterator, IncrementalCompact, Pair].args
  @eval begin
    Base.start(x::NI.$T) = NI.start(x)
    Base.next(x::NI.$T, st) = NI.next(x, st)
    Base.done(x::NI.$T, st) = NI.done(x, st)
  end
end

PhiNode(x, y) = PhiNode(Any[x...], Any[y...])

CFG(bs) = CFG(bs, map(b -> b.stmts.first, bs[2:end]))

Base.show(io::IO, x::SSAValue) = print(io, "%", x.id)

Base.show(io::IO, x::Argument) = print(io, "%%", x.n)

function _compact!(code::IRCode)
  compact = IncrementalCompact(code)
  state = start(compact)
  while !done(compact, state)
    _, state = next(compact, state)
  end
  return finish(compact), compact.ssa_rename
end

rename(x, m) = x
rename(x::SSAValue, m) = m[x.id]
rename(xs::AbstractVector, m) = map(x -> rename(x, m), xs)

function usages(ir, xs)
  us = Dict(x => [] for x in xs)
  for i = 1:length(ir.stmts), u in userefs(ir.stmts[i])
    u[] ∈ xs && push!(us[u[]], SSAValue(i))
  end
  return us
end

function code_ir(f, T)
  ci = code_typed(f, T, optimize=false)[1][1]
  ssa = compact!(NI.just_construct_ssa(ci, copy(ci.code), length(T.parameters), [NI.NullLineInfo]))
end

function code_irm(ex)
  isexpr(ex, :call) || error("@code_ir f(args...)")
  f, args = ex.args[1], ex.args[2:end]
  :(code_ir($(esc(f)), typesof($(esc.(args)...))))
end

macro code_ir(ex)
  code_irm(ex)
end

function blockidx(ir, i::Integer)
  i = findlast(x -> x <= i, ir.cfg.index)
  i == nothing ? 1 : i+1
end

blockidx(ir, i::SSAValue) = blockidx(ir, i.id)

# Dominance frontiers

function domfront(cfg, dt = construct_domtree(cfg))
  fronts = [Set{Int}() for _ in cfg.blocks]
  for b = 1:length(cfg.blocks)
    length(cfg.blocks[b].preds) >= 2 || continue
    for p in cfg.blocks[b].preds
      runner = p
      while runner != dt.idoms[b]
        runner == b && break
        push!(fronts[runner], b)
        runner = dt.idoms[runner]
      end
    end
  end
  return fronts
end
