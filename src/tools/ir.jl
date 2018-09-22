import Core: SSAValue, GotoNode, Compiler
import Core: Typeof
import Core.Compiler: CodeInfo, IRCode, CFG, BasicBlock, Argument, ReturnNode,
  NullLineInfo, just_construct_ssa, compact!, NewNode, InferenceState, OptimizationState,
  GotoIfNot, PhiNode, PiNode, StmtRange, IncrementalCompact, insert_node!, insert_node_here!,
  compact!, finish, DomTree, construct_domtree, dominates, userefs, widenconst, types, verify_ir
using InteractiveUtils: typesof

afterphi(ir, loc) = ir.stmts[loc] isa PhiNode ? afterphi(ir, loc+1) : loc

insert_blockstart!(ir::IRCode, pos, typ, val) =
  insert_node!(ir, afterphi(ir, ir.cfg.blocks[pos].stmts[1]), typ, val)

function insert_blockend!(ir::IRCode, pos, typ, val)
  i = first(ir.cfg.blocks[pos].stmts)
  j = last(ir.cfg.blocks[pos].stmts)
  if !(ir.stmts[j] isa Union{GotoNode,GotoIfNot,ReturnNode})
    return insert_node!(ir, j, typ, val, true)
  end
  while j > i && ir.stmts[j-1] isa Union{GotoNode,GotoIfNot,ReturnNode}
    j -= 1
  end
  insert_node!(ir, j, typ, val)
end

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

exprtype(ir::IRCode, x::Argument) = widenconst(ir.argtypes[x.n])
exprtype(ir::IRCode, x::SSAValue) = widenconst(types(ir)[x])
exprtype(ir::IRCode, x::GlobalRef) = isconst(x.mod, x.name) ? Typeof(getfield(x.mod, x.name)) : Any
exprtype(ir::IRCode, x::QuoteNode) = Typeof(x.value)
# probably can fall back to any here
exprtype(ir::IRCode, x::Union{Type,Number,Nothing,Tuple,Function,Val}) = Typeof(x)
exprtype(ir::IRCode, x::Expr) = error(x)

rename(x, m) = x
rename(x::SSAValue, m) = m[x.id]
rename(xs::AbstractVector, m) = map(x -> rename(x, m), xs)
rename(xs::AbstractSet, m) = Set(rename(x, m) for x in xs)
rename(d::AbstractDict, m) = Dict(k => rename(v, m) for (k, v) in d)

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

Base.range(b::BasicBlock) = b.stmts.start:b.stmts.stop

xcall(mod::Module, f::Symbol, args...) = Expr(:call, GlobalRef(mod, f), args...)
xcall(f::Symbol, args...) = xcall(Base, f, args...)

const unreachable = ReturnNode()

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
