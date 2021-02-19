# Utilities

Zygote's gradients can be used to construct a Jacobian (by repeated evaluation)
or a Hessian (by taking a second derivative).

```@docs
Zygote.jacobian
Zygote.hessian
```

Zygote also provides a set of helpful utilities. These are all "user-level" tools –
in other words you could have written them easily yourself, but they live in
Zygote for convenience.

```@docs
Zygote.@showgrad
Zygote.hook
Zygote.dropgrad
Zygote.Buffer
Zygote.forwarddiff
Zygote.ignore
Zygote.checkpointed
```

`Params` and `Grads` can be copied to and from arrays using the `copy!` function.

### Operations with Grads

Map and broadcast are supported for the dictionary-like `Grads` object.
```julia
using Zygote, Test

w = rand(2)
x1 = rand(2)
x2 = rand(2)
b = rand(2)

gs1 = gradient(() -> sum(w .* x1 .+ b), Params([w])) 
gs2 = gradient(() -> sum(w .* x2  .+ b), Params([w])) 

# accumulate gradients
gs = gs .+ gs
@test gs[w] ≈ gs1[w] + gs2[w] 
@test gs[b] ≈ gs1[b] + gs2[b] 

# clip gradients in-place
map!(x -> clamp!(x, -0.1, 0.1), gs, gs)
```
