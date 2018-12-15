using IRTools
using IRTools: IR, slots!, argnames!, varargs!, xcall, SSAValue
using ..Zygote: Argument, argmap, ignored, isexpr

function record!(ir::IR)
  ir = map(ex -> argmap(a -> Argument(a.n+2), ex), ir)
  grads = Dict()
  grad(x::Union{SSAValue,Argument}) = grads[x]
  grad(x) = nothing
  ks = keys(ir)
  for n = 1:length(ir.args)
    grads[Argument(n+2)] = pushfirst!(ir, xcall(Base, :getindex, Argument(2)))
  end
  for k in ks
    ex = ir[k].expr
    if isexpr(ex, :call) && !ignored(ir, ex)
      Δ = insert!(ir, k, xcall(Base, :tuple, grad.(ex.args)...))
      ir[k] = xcall(Forward, :_forward, Δ, ex.args...)
    end
  end
  return ir
end

@generated function _forward(Δ::Tuple, f, args...)
  T = Tuple{f,args...}
  meta = IRTools.meta(T)
  forw = record(IR(meta))
  pushfirst!(forw.args, Any, Any)
  argnames!(meta, Symbol("#self#"), :Δ, :f, :args)
  forw = varargs!(meta, forw, 3)
  forw = slots!(forw)
  return IRTools.update!(meta, forw)
end
