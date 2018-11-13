import .OrdinaryDiffEq: ODEProblem, solve, remake
using ForwardDiff: Dual

@grad function OrdinaryDiffEq.solve(prob::ODEProblem, args...; kw...)
  ps = prob.p
  ps = [Dual(p, [i==j for j = 1:length(ps)]...) for (i, p) in enumerate(ps)]
  prob = remake(prob, u0 = convert.(eltype(ps), prob.u0), p = ps)
  sol = solve(prob, args...; kw...)
  sol_ = OrdinaryDiffEq.DiffEqBase.build_solution(
    sol.prob,sol.alg,sol.t,map(x -> ForwardDiff.value.(x), sol.u),
    dense = sol.dense, k = sol.k, interp = sol.interp, retcode = sol.retcode)
  sol_, function (dsol)
    dps = map(i -> sum(ForwardDiff.partials.(sol, i) .* dsol), 1:length(ps))
    ((nt_nothing(prob)...,p=dps), map(_ -> nothing, args)...)
  end
end
