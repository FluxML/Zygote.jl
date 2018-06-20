import Core: SSAValue, GotoNode, Compiler
import Core.Compiler: IRCode, CFG, BasicBlock, Argument, ReturnNode,
  NullLineInfo, just_construct_ssa, compact!, NewNode,
  GotoIfNot, PhiNode, PiNode, StmtRange, IncrementalCompact, insert_node!, insert_node_here!,
  compact!, finish, DomTree, construct_domtree, dominates, userefs
using InteractiveUtils: typesof

for T in :[IRCode, IncrementalCompact, Compiler.UseRef, Compiler.UseRefIterator, Compiler.TypesView].args
  @eval begin
    Base.getindex(ir::$T, a...) = Compiler.getindex(ir, a...)
    Base.setindex!(ir::$T, a...) = Compiler.setindex!(ir, a...)
  end
end

Base.getindex(r::StmtRange, i) = (r.first:r.last)[i]

for T in :[UseRefIterator, IncrementalCompact, Pair].args
  @eval Base.iterate(x::Compiler.$T, a...) = Compiler.iterate(x, a...)
end

PhiNode(x, y) = PhiNode(Any[x...], Any[y...])

CFG(bs) = CFG(bs, map(b -> b.stmts.first, bs[2:end]))

function finish_dc(ic::IncrementalCompact)
  Compiler.non_dce_finish!(ic)
  return Compiler.complete(ic)
end

function _compact!(code::IRCode)
    compact = IncrementalCompact(code)
    foreach(x -> nothing, compact)
    return finish_dc(compact), compact.ssa_rename
end

function argmap(f, @nospecialize(stmt))
    urs = userefs(stmt)
    for op in urs
        val = op[]
        if isa(val, Argument)
            op[] = f(val)
        end
    end
    return urs[]
end

exprtype(ir::IRCode, x::Argument) = ir.argtypes[x.n]
exprtype(ir::IRCode, x::SSAValue) = Compiler.types(ir)[x]
exprtype(ir::IRCode, x::GlobalRef) = isconst(x.mod, x.name) ? typeof(getfield(x.mod, x.name)) : Any
exprtype(ir::IRCode, x::QuoteNode) = typeof(x.value)
# probably can fall back to any here
exprtype(ir::IRCode, x::Union{Type,Number}) = Core.Typeof(x)

rename(x, m) = x
rename(x::SSAValue, m) = m[x.id]
rename(xs::AbstractVector, m) = map(x -> rename(x, m), xs)
rename(xs::AbstractSet, m) = map(x -> rename(x, m), xs)

function usages(ir)
  us = Dict()
  for i = 1:length(ir.stmts), u in userefs(ir.stmts[i])
    push!(get!(us, u[], []), SSAValue(i))
  end
  return us
end

function blockidx(ir, i::Integer)
  i = findlast(x -> x <= i, ir.cfg.index)
  i == nothing ? 1 : i+1
end

blockidx(ir, i::SSAValue) = blockidx(ir, i.id)

Base.range(b::BasicBlock) = b.stmts.first:b.stmts.last

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
