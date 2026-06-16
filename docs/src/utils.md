```@meta
CollapsedDocStrings = true
```


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
gradient calculation.

```@docs
Zygote.withgradient
Zygote.withjacobian
Zygote.@showgrad
Zygote.hook
Zygote.Buffer
Zygote.forwarddiff
Zygote.checkpointed
Zygote.eager_update!
```
