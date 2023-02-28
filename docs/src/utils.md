# Utilities

Zygote's gradients can be used to construct a Jacobian (by repeated evaluation)
or a Hessian (by taking a second derivative).

```@docs
Zygote.jacobian
Zygote.hessian
Zygote.hessian_reverse
Zygote.diaghessian
```

Zygote also provides a set of helpful utilities. These are all "user-level" tools –
in other words you could have written them easily yourself, but they live in
Zygote for convenience.

See `ChainRules.ignore_derivatives` if you want to exclude some of your code from the
gradient calculation. This replaces previous Zygote-specific `ignore` and `dropgrad`
functionality.

```@docs
Zygote.withgradient
Zygote.withjacobian
Zygote.@showgrad
Zygote.hook
Zygote.Buffer
Zygote.forwarddiff
Zygote.checkpointed
```

`Params` and `Grads` can be copied to and from arrays using the `copy!` function.

## Working with Grads

Map, broadcast, and iteration are supported for the dictionary-like `Grads` objects.
These operations are value based and preserve the keys.

```julia
using Zygote, Test

w, x1, x2, b = rand(2), rand(2), rand(2), rand(2)

gs1 = gradient(() -> sum(tanh.(w .* x1 .+ b)), Params([w, b]))
gs2 = gradient(() -> sum(tanh.(w .* x2 .+ b)), Params([w, b]))

# accumulate gradients
gs = gs1 .+ gs2
@test gs[w] ≈ gs1[w] + gs2[w]
@test gs[b] ≈ gs1[b] + gs2[b]

# gradients and IdDict interact nicely
# note that an IdDict must be used for gradient algebra on the GPU
gs .+= IdDict(p => randn(size(p)) for p in keys(gs))

# clip gradients
map(x -> clamp.(x, -0.1, 0.1), gs)

# clip gradients in-place
foreach(x -> clamp!(x, -0.1, 0.1), gs)

for (p, g) in pairs(gs)
  # do something with parameter `p` and corresponding gradient `g`
end

# note that gradients must be w.r.t. to the same parameter key set
gs3 = gradient(() -> sum(tanh.(w .* x2)), Params([w]))
# gs3 does not have the key b
@test_throws ArgumentError gs1 .+ gs3
```
