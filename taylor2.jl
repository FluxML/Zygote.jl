using Zygote, Flux, TaylorSeries, TaylorIntegration
using TaylorSeries: constant_term, derivative
using TaylorIntegration: jetcoeffs!
using Zygote: Context, @showgrad, _forward

u0 = Float32[2.; 0.]
ord = 6

dynamics_net = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))

dynamics_net(Taylor1.(u0, 3))

gradient(0.) do t
    t_tay = Taylor1(typeof(t),ord) + t
    u_tay = Taylor1.(u0, ord)
    du = similar(u_tay)
    uaux = similar(u_tay)
    jetcoeffs!(t_tay,u_tay,du,uaux) do t, u, du
        du .= map(sin, u)
        # du[:] = map(sin, u)
    end
    return u_tay[1][0]
end
