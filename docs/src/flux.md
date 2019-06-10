# Flux

It's easy to use Zygote in place of Flux's default AD, Tracker, just by changing `Tracker.gradient` to `Zygote.gradient`. The API is otherwise the same.

```julia
julia> using Flux, Zygote

julia> m = Chain(Dense(10, 5, relu), Dense(5, 2))
Chain(Dense(10, 5, NNlib.relu), Dense(5, 2))

julia> x = rand(10);

julia> gs = gradient(() -> sum(m(x)), params(m))
Grads(...)

julia> gs[m[1].W]
5×10 Array{Float32,2}:
 -0.255175  -1.2295   ...
```

You can use optimisers and update gradients as usual.

```julia
julia> opt = ADAM();

julia> Flux.Optimise.update!(opt, params(m), gs)
```
