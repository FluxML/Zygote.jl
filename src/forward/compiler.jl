using IRTools.All
using IRTools: Pipe
using Base: tail

ntail(x, n) = n <= 0 ? x : xcall(:tail, ntail(x, n-1))

function instrument!(pr, v, st)
  ex = st.expr
  if isexpr(ex, :new)
    st = stmt(st, expr = xcall(Zygote, :__new__, ex.args...))
    pr[v] = st
  elseif isexpr(ex, :splatnew)
    st = stmt(st, expr = xcall(Zygote, :__splatnew__, ex.args...))
    pr[v] = st
  end
  return st
end

function dual(ir)
  args = copy(arguments(ir))
  dx = argument!(ir, at = 1)
  Δs = Dict()
  for bl in blocks(ir)[2:end], arg in copy(arguments(bl))
    Δs[arg] = argument!(bl, insert = false)
  end
  pr = Pipe(ir)
  partial(x::Variable) = Δs[x]
  partial(x) = push!(pr, xcall(Forward, :zerolike, x))
  partial(v, x::Variable) = Δs[x]
  partial(v, x) = insert!(pr, v, xcall(Forward, :zerolike, x))
  for (i, x) in enumerate(args)
    if i == length(args) && ir.meta.method.isva
      Δs[x] = push!(pr, ntail(dx, i-1))
    else
      Δs[x] = push!(pr, xcall(:getindex, dx, i))
    end
  end
  branches(pr) do br
    args = arguments(br)
    if isreturn(br)
      args[1] = push!(pr, xcall(:tuple, args[1], partial(args[1])))
    else
      for arg in copy(args)
        push!(args, partial(arg))
      end
    end
    br
  end
  for (v, st) in pr
    st = instrument!(pr, v, st)
    if isexpr(st.expr, :meta, :inbounds, :loopinfo)
      Δs[v] = nothing
    elseif isexpr(st.expr, :boundscheck) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Base, :not_int)) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Core, :(===))) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Main, :(===)))
      Δs[v] = false
    elseif isexpr(st.expr, :call)
      dargs = insert!(pr, v, xcall(:tuple, partial.((v,), st.expr.args)...))
      result = insert!(pr, v, stmt(st, expr = xcall(Forward, :_pushforward, dargs, st.expr.args...)))
      pr[v] = xcall(:getindex, result, 1)
      Δs[v] = push!(pr, xcall(:getindex, result, 2))
    elseif !isexpr(st.expr)
      Δs[v] = push!(pr, xcall(Forward, :zerolike, v))
    else
      error("Unsupported $(st.expr.head) expression")
    end
  end
  ir = finish(pr)
  return ir
end

@dynamo function _pushforward(_, x...)
  ir = IR(x...)
  ir === nothing && return :(error("non-differentiable function $(args[2])"))
  ir = Zygote.instrument(ir)
  ir.meta.code.inlineable = true
  return dual(ir)
end
