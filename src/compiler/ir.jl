import Core: SSAValue, GotoNode, Compiler
import Core.Compiler: IRCode, CFG, BasicBlock, Argument, ReturnNode,
  NullLineInfo, just_construct_ssa, compact!, NewNode,
  GotoIfNot, PhiNode, PiNode, StmtRange, IncrementalCompact, insert_node!, insert_node_here!,
  compact!, finish, DomTree, construct_domtree, dominates, userefs
using InteractiveUtils: typesof

for T in :[IRCode, IncrementalCompact, Compiler.UseRef, Compiler.UseRefIterator].args
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

function code_ir(f, T)
  ci = code_typed(f, T, optimize=false)[1][1]
  ssa = compact!(just_construct_ssa(ci, copy(ci.code), Int(which(f, T).nargs)-1))
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
