import Core: SSAValue, GotoNode, Compiler
import Core: Typeof
import Core.Compiler: CodeInfo, IRCode, CFG, BasicBlock, Argument, ReturnNode,
  NullLineInfo, just_construct_ssa, compact!, NewNode, InferenceState, OptimizationState,
  GotoIfNot, PhiNode, PiNode, StmtRange, IncrementalCompact, insert_node!, insert_node_here!,
  compact!, finish, DomTree, construct_domtree, dominates, userefs, widenconst, types, verify_ir
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

exprtype(ir::IRCode, x::Argument) = widenconst(ir.argtypes[x.n])
exprtype(ir::IRCode, x::SSAValue) = widenconst(types(ir)[x])
exprtype(ir::IRCode, x::GlobalRef) = isconst(x.mod, x.name) ? Typeof(getfield(x.mod, x.name)) : Any
exprtype(ir::IRCode, x::QuoteNode) = Typeof(x.value)
# probably can fall back to any here
exprtype(ir::IRCode, x::Union{Type,Number,Nothing,Tuple{},Function}) = Typeof(x)
exprtype(ir::IRCode, x::Expr) = error(x)

rename(x, m) = x
rename(x::SSAValue, m) = m[x.id]
rename(xs::AbstractVector, m) = map(x -> rename(x, m), xs)
rename(xs::AbstractSet, m) = Set(rename(x, m) for x in xs)

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

# SSA contruction (forked from Base for untyped code)

import Core.Compiler: normalize, strip_trailing_junk!, compute_basic_blocks,
  scan_slot_def_use, LineInfoNode, construct_ssa!, IR_FLAG_INBOUNDS

function just_construct_ssa(ci::CodeInfo, code::Vector{Any}, nargs::Int, sp)
  ci.ssavaluetypes = Any[Any for _ = 1:length(code)]
  slottypes = Any[Any for _ = 1:length(ci.slotnames)]
  inbounds_depth = 0 # Number of stacked inbounds
  meta = Any[]
  flags = fill(0x00, length(code))
  for i = 1:length(code)
    stmt = code[i]
    if isexpr(stmt, :inbounds)
      arg1 = stmt.args[1]
      if arg1 === true # push
        inbounds_depth += 1
      elseif arg1 === false # clear
        inbounds_depth = 0
      elseif inbounds_depth > 0 # pop
        inbounds_depth -= 1
      end
      stmt = nothing
    else
      stmt = normalize(stmt, meta)
    end
    code[i] = stmt
    if !(stmt === nothing)
      if inbounds_depth > 0
        flags[i] |= IR_FLAG_INBOUNDS
      end
    end
  end
  strip_trailing_junk!(ci, code, flags)
  cfg = compute_basic_blocks(code)
  defuse_insts = scan_slot_def_use(nargs, ci, code)
  domtree = construct_domtree(cfg)
  ir = let code = Any[nothing for _ = 1:length(code)]
    argtypes = slottypes[1:(nargs+1)]
    IRCode(code, Any[], ci.codelocs, flags, cfg, collect(LineInfoNode, ci.linetable), argtypes, meta, sp)
  end
  ir = construct_ssa!(ci, code, ir, domtree, defuse_insts, nargs, sp, slottypes)
  return ir
end
