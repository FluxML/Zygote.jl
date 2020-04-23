# Interoperability with Optim.jl

function optim_funs(loss, ps::Params)
    function f(w)
        copyto!(ps, w)
        loss()
    end
    
    function g!(G, w)
        copyto!(ps, w)
        gs = gradient(loss, ps)
        copyto!(G, gs)
    end
    
    function fg!(F, G, w)
        copyto!(ps, w)
        l, back = pullback(loss, ps)        
        if G !== nothing
            gs = back(1)
            copyto!(G, gs)
        end
        return l
    end

    x0 = vcat((vec(p) for p in ps)...)
    
    f, g!, fg!, x0
end
