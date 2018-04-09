Base.show(io::IO, x::SSAValue) = print(io, "%", x.id)

struct Argument
  n::Int
end

Base.show(io::IO, x::Argument) = print(io, "%%", x.n)

struct Phi
  edges::Vector{Int}
  values::Vector{Any}
end

function Base.show(io::IO, phi::Phi)
  print(io, "ϕ(")
  join(io, ["$edge => $value" for (edge,value) in zip(phi.edges,phi.values)], ", ")
  print(io, ")")
end

struct BasicBlock
  succs::Vector{Int}
  preds::Vector{Int}
  first::Int
  last ::Int
end

Base.range(b::BasicBlock) = b.first:b.last

struct ControlFlowGraph
  blocks::Vector{BasicBlock}
  index::Vector{Int}
end

const CFG = ControlFlowGraph

struct IRCode
  stmts::Vector{Any}
  cfg::ControlFlowGraph
end

function Base.show(io::IO, code::IRCode)
  println(io, "IRCode")
  for (i, x) in enumerate(code.stmts)
    blk = findfirst([1, code.cfg.index...], i)
    blk == 0 || println(io, "$blk:")
    print(io, lpad(i, 3), " | ")
    println(io, x)
  end
end

struct Block
  ir::IRCode
  n::Int
end

BasicBlock(b::Block) = b.ir.cfg.blocks[b.n]
Base.range(b::Block) = range(BasicBlock(b))

# IR manipulation

function bumpcfg!(ir, idx)
  bi = findlast(x -> x ≤ idx, ir.cfg.index)+1
  b = ir.cfg.blocks[bi]
  ir.cfg.blocks[bi] = BasicBlock(b.succs, b.preds, b.first, b.last+1)
  @show ir.cfg.blocks[bi]
  for i = bi+1:length(ir.cfg.blocks)
    ir.cfg.index[i-1] += 1
    b = ir.cfg.blocks[i]
    ir.cfg.blocks[i] = BasicBlock(b.succs, b.preds, b.first+1, b.last+1)
  end
end

bumpssa(x) = x
bumpssa(x::SSAValue) = SSAValue(x.id+1)
bumpssa(p::Phi) = Phi(p.edges, bumpssa.(p.values))

function bumpssa(x::Expr)
  y = Expr(x.head, bumpssa.(x.args)...)
  y.typ = x.typ
  return y
end

function bumpssa!(ir, idx)
  for i = idx+1:length(ir.stmts)
    ir.stmts[i] = bumpssa(ir.stmts[i])
  end
end

function Base.insert!(ir::IRCode, idx, x)
  insert!(ir.stmts, idx, x)
  bumpcfg!(ir, idx)
  bumpssa!(ir, idx)
  return ir
end

Base.push!(ir::IRCode, x) = insert!(ir, length(ir.stmts)+1, x)

Base.insert!(b::Block, idx, x) = insert!(b.ir, range(b)[1]+idx-1, x)

# Load IR from JSON

function jsonblock(blk)
  BasicBlock(blk["succs"], blk["preds"], blk["stmts"]...)
end

function jsoncfg(cfg)
  CFG(jsonblock.(cfg["blocks"]), cfg["index"])
end

function jsonexpr(x::Associative)
  if haskey(x, "head")
    ex = Expr(Symbol(x["head"]), jsonexpr.(x["args"])...)
    ex.head == :ref && return GlobalRef(getfield(Main, Symbol(ex.args[1])), Symbol(ex.args[2]))
    ex
  elseif haskey(x, "id")
    return SSAValue(x["id"])
  elseif haskey(x, "n")
    return Argument(x["n"])
  elseif haskey(x, "val")
    Expr(:return, jsonexpr(x["val"]))
  elseif haskey(x, "cond")
    Expr(:gotoifnot, jsonexpr(x["cond"]), x["dest"])
  elseif haskey(x, "label")
    GotoNode(x["label"])
  elseif haskey(x, "edges")
    Phi(x["edges"], jsonexpr.(x["values"]))
  else
    x
  end
end

jsonexpr(x) = x

function jsonir(ir)
  IRCode(jsonexpr.(ir["stmts"]), jsoncfg(ir["cfg"]))
end
