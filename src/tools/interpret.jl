mutable struct Interpreter
  ir::IRCode
  args::Vector{Any}
  locals::Dict{SSAValue,Any}
  pc::Int
  Interpreter(ir::IRCode, args...) = new(ir, Any[args...], Dict(), 1)
end

blockidx(i::Interpreter) = blockidx(i.ir, i.pc)
assign!(i::Interpreter, x) = (i.locals[SSAValue(i.pc)] = lookup(i, x))

lookup(i::Interpreter, x) = x
lookup(i::Interpreter, x::QuoteNode) = x.value
lookup(i::Interpreter, x::GlobalRef) = getfield(x.mod, x.name)
lookup(i::Interpreter, x::Argument) = i.args[x.n]
lookup(i::Interpreter, x::SSAValue) = i.locals[x]
lookup(i::Interpreter, x::Expr) =
  error("Can't lookup $(x.head) expr")

function phis!(i::Interpreter, blk)
  while (phi = i.ir.stmts[i.pc]) isa PhiNode
    assign!(i, phi.values[findfirst(e -> e == blk, phi.edges)])
    i.pc += 1
  end
end

call(f, xs...) = f(xs...)

function step!(i::Interpreter)
  blk = blockidx(i)
  ex = i.ir.stmts[i.pc]
  if ex isa GotoNode
    i.pc = i.ir.cfg.index[ex.label-1]
  elseif ex isa GotoIfNot
    lookup(i, ex.cond) ? (i.pc += 1) : (i.pc = i.ir.cfg.index[ex.dest-1])
  elseif ex isa ReturnNode
    return lookup(i, ex.val)
  elseif ex == nothing
    i.pc += 1
  elseif isexpr(ex, :call)
    assign!(i, call(lookup.(Ref(i),ex.args)...))
    i.pc += 1
  elseif isexpr(ex, GlobalRef)
    assign!(i, getfield(ex.mod, ex.name))
    i.pc += 1
  elseif ex isa PiNode
    assign!(i, lookup(i, ex.val))
    i.pc += 1
  else
    error("can't handle $ex")
  end
  phis!(i, blk)
  return
end

function run!(i::Interpreter)
  while !(i.ir.stmts[i.pc] isa ReturnNode)
    step!(i)
  end
  return lookup(i, i.ir.stmts[i.pc].val)
end

interpret(ir, args...) = run!(Interpreter(ir, args...))
