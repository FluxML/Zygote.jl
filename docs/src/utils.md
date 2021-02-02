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
