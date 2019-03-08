# Custom Adjoints

## Pullbacks

`gradient` is really just syntactic sugar around the more fundamental function `forward`.

```julia
julia> y, back = Zygote.forward(sin, 0.5);

julia> y
0.479425538604203
```

`forward` gives two outputs: the result of the original function, `sin(0.5)`, and a *pullback*, here called `back`. `back` implements the gradient computation for `sin`, accepting a derivative and producing a new one. In mathematical terms, it implements a vector-Jacobian product. Where ``y = f(x)`` and the gradient ``\frac{\partial l}{\partial x}`` is written ``\bar{x}``, the pullback ``\mathcal{B}_y`` computes:

```math
\bar{x} = \frac{\partial l}{\partial x} = \frac{\partial l}{\partial y} \frac{\partial y}{\partial x} = \mathcal{B}_y(\bar{y})
```

To make this concrete, take the function ``y = \sin(x)``. ``\frac{\partial y}{\partial x} = \cos(x)``, so the pullback is ``\bar{y} \cos(x)``. In other words `forward(sin, x)` behaves the same as

```julia
dsin(x) = sin(x), ȳ -> (ȳ * cos(x),)
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

```julia
julia> function mygradient(f, x...)
         _, back = Zygote.forward(f, x...)
         back(1)
       end
mygradient (generic function with 1 method)

julia> mygradient(sin, 0.5)
(0.8775825618903728,)
```

## Custom Adjoints

We can extend Zygote to a new function with the `@adjoint` function.

```julia
julia> mul(a, b) = a*b

julia> using Zygote: @adjoint

julia> @adjoint mul(a, b) = mul(a, b), c̄ -> (c̄*b, c̄*a)

julia> gradient(mul, 2, 3)
(3, 2)
```

It might look strange that we write `mul(a, b)` twice here. In this case we want to call the normal `mul` function for the forward pass, but you may also want to modify the forward pass (for example, to capture intermediate results in the pullback).

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
(x = 0.5547001962252291, y = 0.8320502943378437)
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

```julia
julia> hook(f, x) = x
hook (generic function with 1 method)

julia> @adjoint hook(f, x) = x, x̄ -> (nothing, f(x̄))
```

`hook` doesn't seem that interesting, as it doesn't do anything. But the fun part is in the adjoint; it's allowing us to apply a function `f` to the gradient of `x`.

```julia
julia> gradient((a, b) -> hook(-, a)*b, 2, 3)
(-3, 2)
```

We could use this for debugging or modifying gradients (e.g. gradient clipping).

```julia
julia> gradient((a, b) -> hook(ā -> @show(ā), a)*b, 2, 3)
ā = 3
(3, 2)
```

Zygote provides both `hook` and `@showgrad` so you don't have to write these yourself.

### Checkpointing

A more advanced example is checkpointing, in which we save memory by re-computing the forward pass of a function during the backwards pass. To wit:

```julia
julia> checkpoint(f, x) = f(x)
checkpoint (generic function with 1 method)

julia> @adjoint checkpoint(f, x) = f(x), ȳ -> Zygote._forward(f, x)[2](ȳ)

julia> gradient(x -> checkpoint(sin, x), 1)
(0.5403023058681398,)
```

If a function has side effects we'll see that the forward pass happens twice, as expected.

```julia
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
