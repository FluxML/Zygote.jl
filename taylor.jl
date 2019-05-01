using Zygote, Flux, TaylorSeries, TaylorIntegration
using TaylorSeries: constant_term, derivative
using TaylorIntegration: jetcoeffs!
using Zygote: @showgrad

taylorshift(x, n) = x + Taylor1(typeof(x), n)

derive(x) = derivative(x)[0]

function taylor_highify(f,u,p,t; ord=6)
    T = typeof(u[1])
    t_tay = Taylor1(T,ord) .+ t
    dof = length(u)
    u_tay = Array{Taylor1{T}}(undef,dof)
    du = Array{Taylor1{T}}(undef,dof)
    uaux = Array{Taylor1{T}}(undef,dof)
    for i in eachindex(u)
        u_tay[i] = Taylor1(u[i],ord)
    end
    jetcoeffs!((t,u,du) -> f(du,u,p,t),t_tay,u_tay,du,uaux)
    return u_tay
end

nfe = 0
dynamics_net = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
p = nothing
t = nothing
function model(u,p,t)
    global nfe += 1
    return dynamics_net(u)
end

function dudt(du,u,p,t)
    du .= Flux.data(model(u,p,t))
end

u0 = Float32[2.; 0.]

f = t -> sum(derive.(taylor_highify(dudt,u0,p,t)))

f(0.)

gradient(f, 0.)
