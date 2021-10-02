# Performance Tips

## getproperty

Zygote has known performance problems with `getproperty`.
In particular, if you try to differentiate something like
```julia
struct Foo
    x::Float64
end

Zygote.gradient(x -> x.x, Foo(5.0))
```
you will wind that it is not type-stable, despite the function being differentiated being
type-stable.

See the docstring for the function `pullback_for_default_literal_getproperty` in
[ZygoteRules.jl](https://github.com/FluxML/ZygoteRules.jl) for instructions on how to
work around this problem.

[This PR](https://github.com/FluxML/Zygote.jl/pull/909) may make this workaround redundant
at some point.
