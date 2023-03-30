# Custom Adjoints

!!! note "Prefer to use ChainRulesCore to define custom adjoints"
    Zygote supports the use of [ChainRulesCore](http://www.juliadiff.org/ChainRulesCore.jl/stable/) to define custom sensitivities.
    It is preferred to define the custom sensitivities using `ChainRulesCore.rrule` as they will work for many AD systems, not just Zygote.
    These sensitivities can be added in your own package, or for Base/StdLib functions they can be added to [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/).
    To define custom sensitivities using ChainRulesCore, define a `ChainRulesCore.rrule(f, args...; kwargs...)`. Head to [ChainRules project's documentation](https://www.juliadiff.org/ChainRulesCore.jl/stable/) for more information.
    **If you are defining your custom adjoints using ChainRulesCore then you do not need to read this page**, and can consider it as documenting a legacy feature.

    This page exists to describe how Zygote works, and how adjoints can be directly defined for Zygote.
    Defining adjoints this way does not make them accessible to other AD systems, but does let you do things that directly depend on how Zygote works.
    It allows for specific definitions of adjoints that are only defined for Zygote (which might work differently to more generic definitions defined for all AD).


The `@adjoint` macro is an important part of Zygote's interface; customising your backwards pass is not only possible but widely used and encouraged. While there are specific utilities available for common things like gradient clipping, understanding adjoints will give you the most flexibility. We first give a bit more background on what these pullback things are.

## Pullbacks

`gradient` is really just syntactic sugar around the more fundamental function `pullback`.

```jldoctest adjoints
julia> using Zygote

julia> y, back = Zygote.pullback(sin, 0.5);

julia> y
0.479425538604203
```

`pullback` gives two outputs: the result of the original function, `sin(0.5)`, and a *pullback*, here called `back`. `back` implements the gradient computation for `sin`, accepting a derivative and producing a new one. In mathematical terms, it implements a vector-Jacobian product. Where ``y = f(x)`` and the gradient ``\frac{\partial l}{\partial x}`` is written ``\bar{x}``, the pullback ``\mathcal{B}_y`` computes:

```math
\bar{x} = \frac{\partial l}{\partial x} = \frac{\partial l}{\partial y} \frac{\partial y}{\partial x} = \mathcal{B}_y(\bar{y})
```

To make this concrete, take the function ``y = \sin(x)``. ``\frac{\partial y}{\partial x} = \cos(x)``, so the pullback is ``\bar{y} \cos(x)``. In other words `pullback(sin, x)` behaves the same as

```julia
dsin(x) = (sin(x), ȳ -> (ȳ * cos(x),))
```

`gradient` takes a function ``l = f(x)`` and assumes ``l̄ = \frac{\partial l}{\partial l} = 1`` and feeds this in to the pullback. In the case of `sin`,

```julia
julia> function gradsin(x)
         _, back = dsin(x)
         back(1)
       end
gradsin (generic function with 1 method)

julia> gradsin(0.5)
(0.8775825618903728,)

julia> cos(0.5)
0.8775825618903728
```

More generally

```jldoctest adjoints
julia> function mygradient(f, x...)
         _, back = Zygote.pullback(f, x...)
         back(1)
       end
mygradient (generic function with 1 method)

julia> mygradient(sin, 0.5)
(0.8775825618903728,)
```

The rest of this section contains more technical detail. It can be skipped if you only need an intuition for pullbacks; you generally won't need to worry about it as a user.

If ``x`` and ``y`` are vectors, ``\frac{\partial y}{\partial x}`` becomes a Jacobian. Importantly, because we are implementing reverse mode we actually left-multiply the Jacobian, i.e. `v'J`, rather than the more usual `J*v`. Transposing `v` to a row vector and back `(v'J)'` is equivalent to `J'v` so our gradient rules actually implement the *adjoint* of the Jacobian. This is relevant even for scalar code: the adjoint for `y = sin(x)` is `x̄ = cos(x)'*ȳ`; the conjugation is usually moot but gives the correct behaviour for complex code. "Pullbacks" are therefore sometimes called "vector-Jacobian products" (VJPs), and we refer to the reverse mode rules themselves as "adjoints".

Zygote has many adjoints for non-mathematical operations such as for indexing and data structures. Though these can still be seen as linear functions of vectors, it's not particularly enlightening to implement them with an actual matrix multiply. In these cases it's easiest to think of the adjoint as a kind of inverse. For example, the gradient of a function that takes a tuple to a struct (e.g. `y = Complex(a, b)`) will generally take a struct to a tuple (`(ȳ.re, ȳ.im)`). The gradient of a `getindex` `y = x[i...]` is a `setindex!` `x̄[i...] = ȳ`, etc.

## Custom Adjoints

We can extend Zygote to a new function with the `@adjoint` function.

```jldoctest adjoints
julia> mul(a, b) = a*b;

julia> using Zygote: @adjoint

julia> @adjoint mul(a, b) = mul(a, b), c̄ -> (c̄*b, c̄*a)

julia> gradient(mul, 2, 3)
(3.0, 2.0)
```

It might look strange that we write `mul(a, b)` twice here. In this case we want to call the normal `mul` function for the pullback pass, but you may also want to modify the pullback pass (for example, to capture intermediate results in the pullback).

## Custom Types

One good use for custom adjoints is to customise how your own types behave during differentiation. For example, in our `Point` example we noticed that the adjoint is a named tuple, rather than another point.

```julia
import Base: +, -

struct Point
  x::Float64
  y::Float64
end

width(p::Point) = p.x
height(p::Point) = p.y

a::Point + b::Point = Point(width(a) + width(b), height(a) + height(b))
a::Point - b::Point = Point(width(a) - width(b), height(a) - height(b))
dist(p::Point) = sqrt(width(p)^2 + height(p)^2)
```

```julia
julia> gradient(a -> dist(a), Point(1, 2))[1]
(x = 0.4472135954999579, y = 0.8944271909999159)
```

Fundamentally, this happens because of Zygote's default adjoint for `getfield`.

```julia
julia> gradient(a -> a.x, Point(1, 2))
((x = 1, y = nothing),)
```

We can overload this by modifying the getters `height` and `width`.

```julia
julia> @adjoint width(p::Point) = p.x, x̄ -> (Point(x̄, 0),)

julia> @adjoint height(p::Point) = p.y, ȳ -> (Point(0, ȳ),)

julia> Zygote.refresh() # currently needed when defining new adjoints

julia> gradient(a -> height(a), Point(1, 2))
(Point(0.0, 1.0),)

julia> gradient(a -> dist(a), Point(1, 2))[1]
Point(0.4472135954999579, 0.8944271909999159)
```

If you do this you should also overload the `Point` constructor, so that it can handle a `Point` gradient (otherwise this function will error).

```julia
julia> @adjoint Point(a, b) = Point(a, b), p̄ -> (p̄.x, p̄.y)

julia> gradient(x -> dist(Point(x, 1)), 1)
(0.7071067811865475,)
```

## Advanced Adjoints

We usually use custom adjoints to add gradients that Zygote can't derive itself (for example, because they `ccall` to BLAS). But there are some more advanced and fun things we can to with `@adjoint`.

### Gradient Hooks

```jldoctest adjoints
julia> hook(f, x) = x
hook (generic function with 1 method)

julia> @adjoint hook(f, x) = x, x̄ -> (nothing, f(x̄))
```

`hook` doesn't seem that interesting, as it doesn't do anything. But the fun part is in the adjoint; it's allowing us to apply a function `f` to the gradient of `x`.

```jldoctest adjoints
julia> gradient((a, b) -> hook(-, a)*b, 2, 3)
(-3.0, 2.0)
```

We could use this for debugging or modifying gradients (e.g. gradient clipping).

```jldoctest adjoints
julia> gradient((a, b) -> hook(ā -> @show(ā), a)*b, 2, 3)
ā = 3.0
(3.0, 2.0)
```

Zygote provides both `hook` and `@showgrad` so you don't have to write these yourself.

### Checkpointing

A more advanced example is checkpointing, in which we save memory by re-computing the pullback pass of a function during the backwards pass. To wit:

```jldoctest adjoints
julia> checkpoint(f, x) = f(x)
checkpoint (generic function with 1 method)

julia> @adjoint checkpoint(f, x) = f(x), ȳ -> Zygote._pullback(f, x)[2](ȳ)

julia> gradient(x -> checkpoint(sin, x), 1)
(0.5403023058681398,)
```

If a function has side effects we'll see that the pullback pass happens twice, as expected.

```jldoctest adjoints
julia> foo(x) = (println(x); sin(x))
foo (generic function with 1 method)

julia> gradient(x -> checkpoint(foo, x), 1)
1
1
(0.5403023058681398,)
```

### Gradient Reflection

It's easy to check whether the code we're running is currently being differentiated.

```julia
isderiving() = false

@adjoint isderiving() = true, _ -> nothing
```

A more interesting example is to actually detect how many levels of nesting are going on.

```julia
nestlevel() = 0

@adjoint nestlevel() = nestlevel()+1, _ -> nothing
```

Demo:

```julia
julia> function f(x)
         println(nestlevel(), " levels of nesting")
         return x
       end
f (generic function with 1 method)

julia> grad(f, x) = gradient(f, x)[1]
grad (generic function with 1 method)

julia> f(1);
0 levels of nesting

julia> grad(f, 1);
1 levels of nesting

julia> grad(x -> x*grad(f, x), 1);
2 levels of nesting
```
