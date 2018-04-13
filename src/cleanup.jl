function rm_null_grads(ir::IRCode)
  ir = IncrementalCompact(ir)
  dead = []
  isdead(_) = false
  isdead(::Void) = true
  isdead(x::SSAValue) = isdead(ir[x.id])
  for (i, x) in ir
    if !isexpr(x, :call)
      continue
    elseif (x.args[1] == :∇ && isdead(x.args[3])) ||
           (x.args[1] == GlobalRef(Base,:getindex) && isdead(x.args[2])) ||
           (x.args[1] == :accum! && isdead(x.args[3]))
      ir[i] = nothing
    end
  end
  finish(ir)
end

cleanup(ir) = ir |> rm_null_grads |> compact!
