using IRTools: reloop
using IRTools.Inner: Slot
using MacroTools: @q

function primal_slotnames!(ir::IR)
  cnt = 0
  env = Dict()
  rename(x) = prewalk(x -> get(env, x, x), x)
  vs = keys(ir)
  for i = 1:length(vs)
    ex = ir[vs[i]].expr
    if iscall(ex, Zygote, :_pullback)
      v = vs[i+1]
      t = Slot(Symbol(:y, (cnt += 1)))
      ir[v] = :($t = $(ir[v].expr))
      env[v] = t
      v = vs[i+2]
      t = Slot(Symbol(:B, cnt))
      ir[v] = :($t = $(ir[v].expr))
      env[v] = t
    end
  end
  IRTools.prewalk!(x -> get(env, x, x), ir)
  return ir, env
end

function pullback_slotnames!(ir::IR, env)
  IRTools.prewalk!(ir) do x
    x isa Alpha ? env[var(x.id)] : x
  end
end

function slotnames!(adj)
  _, env = primal_slotnames!(adj.primal)
  pullback_slotnames!(adj.adjoint, env)
end

function prettyprint(adj)
  slotnames!(adj)
  ex = @q function adjoint()
      $(reloop(adj.primal, inline = false))
      function back(arg1)
        $(reloop(adj.adjoint, inline = false))
      end
    end
  MacroTools.prettify(ex)
end

macro transform(ex)
  :(prettyprint(Adjoint($(code_irm(ex)))))
end
