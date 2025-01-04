# Zygote

Welcome! Zygote extends the Julia language to support [differentiable programming](https://fluxml.ai/blogposts/2019-02-07-what-is-differentiable-programming/). With Zygote you can write down any Julia code you feel like – including using existing Julia packages – then get gradients and optimise your program. Deep learning, ML and probabilistic programming are all different kinds of differentiable programming that you can do with Zygote.

At least, that's the idea. We're still in beta so expect some adventures.

## Setup

Zygote can be installed from the package manager in Julia's REPL:

```julia
] add Zygote
```

## Taking Gradients

Zygote is easy to understand since, at its core, it has a one-function API (`pullback`), along with a few simple conveniences. Before explaining `pullback`, we'll look at the higher-level function `gradient`.

`gradient` calculates derivatives. For example, the derivative of ``3x^2 + 2x + 1`` is ``6x + 2``, so when `x = 5`, `dx = 32`.

```jldoctest index
julia> using Zygote

julia> gradient(x -> 3x^2 + 2x + 1, 5)
(32.0,)
```

`gradient` returns a tuple, with a gradient for each argument to the function.

```jldoctest index
julia> gradient((a, b) -> a*b, 2, 3)
(3.0, 2.0)
```

This will work equally well if the arguments are arrays, structs, or any other Julia type, but the function should return a scalar (like a loss or objective ``l``, if you're doing optimisation / ML).

```julia
julia> W = rand(2, 3); x = rand(3);

julia> gradient(W -> sum(W*x), W)[1]
2×3 Array{Float64,2}:
 0.0462002  0.817608  0.979036
 0.0462002  0.817608  0.979036

julia> gradient(x -> 3x^2 + 2x + 1, 1//4)
(7//2,)
```

Control flow is fully supported, including recursion.

```jldoctest index
julia> function pow(x, n)
         r = 1
         for i = 1:n
           r *= x
         end
         return r
       end
pow (generic function with 1 method)

julia> gradient(x -> pow(x, 3), 5)
(75.0,)

julia> pow2(x, n) = n <= 0 ? 1 : x*pow2(x, n-1)
pow2 (generic function with 1 method)

julia> gradient(x -> pow2(x, 3), 5)
(75.0,)
```

Data structures are also supported, including mutable ones like dictionaries. Arrays are currently immutable, though [this may change](https://github.com/FluxML/Zygote.jl/pull/75) in future.

```jldoctest index
julia> d = Dict()
Dict{Any, Any}()

julia> gradient(5) do x
         d[:x] = x
         d[:x] * d[:x]
       end
(10.0,)

julia> d[:x]
5
```

## Structs and Types

Julia makes it easy to work with custom types, and Zygote makes it easy to differentiate them. For example, given a simple `Point` type:

```julia
import Base: +, -

struct Point
  x::Float64
  y::Float64
end

a::Point + b::Point = Point(a.x + b.x, a.y + b.y)
a::Point - b::Point = Point(a.x - b.x, a.y - b.y)
dist(p::Point) = sqrt(p.x^2 + p.y^2)
```

```julia
julia> a = Point(1, 2)
Point(1.0, 2.0)

julia> b = Point(3, 4)
Point(3.0, 4.0)

julia> dist(a + b)
7.211102550927978

julia> gradient(a -> dist(a + b), a)[1]
(x = 0.5547001962252291, y = 0.8320502943378437)
```

Zygote's default representation of the "point adjoint" is a named tuple with gradients for both fields, but this can of course be customised too.

This means we can do something very powerful: differentiating through Julia libraries, even if they weren't designed for this. For example, `colordiff` might be a smarter loss function on colours than simple mean-squared-error:

```julia
julia> using Colors

julia> colordiff(RGB(1, 0, 0), RGB(0, 1, 0))
86.60823557376344

julia> gradient(colordiff, RGB(1, 0, 0), RGB(0, 1, 0))
((r = 0.4590887719632896, g = -9.598786801605689, b = 14.181383399012862), (r = -1.7697549557037275, g = 28.88472330558805, b = -0.044793892637761346))
```

## Explicit and Implicit Parameters

It's easy to work with even very large and complex models, and there are few ways to do this. Autograd-style models pass around a collection of weights. Depending on how you write your model, there are multiple ways to *explicitly* take gradients with respect to parameters. For example, the function `linear` accepts the parameters as an argument to the model. So, we directly pass in the parameters, `θ`, as an argument to the function being differentiated.

```@docs
gradient(f, args...)
```

```julia
julia> linear(θ, x) = θ[:W] * x .+ θ[:b]
linear (generic function with 1 method)

julia> x = rand(5);

julia> θ = Dict(:W => rand(2, 5), :b => rand(2))
Dict{Any,Any} with 2 entries:
  :b => [0.0430585, 0.530201]
  :W => [0.923268 … 0.589691]

# Alternatively, use a named tuple or struct rather than a dict.
# θ = (W = rand(2, 5), b = rand(2))

julia> θ̄ = gradient(θ -> sum(linear(θ, x)), θ)[1]
Dict{Any,Any} with 2 entries:
  :b => [1.0, 1.0]
  :W => [0.628998 … 0.433006]
```

We can combine the role of the dictionary and the function here by making a callable struct which
contains the parameters, equivalent to a closure. Passed explicitly to `gradient`, we get a named tuple
with the same field names:

```julia
julia> struct Linear
         W
         b
       end

julia> (l::Linear)(x) = l.W * x .+ l.b

julia> model = Linear(rand(2, 5), rand(2))
Linear([0.267663 … 0.334385], [0.0386873, 0.0203294])

julia> x = rand(5);

julia> dmodel = gradient(model -> sum(model(x)), model)[1]
(W = [0.652543 … 0.683588], b = [1.0, 1.0])
```

Zygote also supports another way to take gradients, via *implicit parameters*. Here the loss function takes zero arguments, but the variables of interest are indicated by a special `Params` object. The function `linear` which depends on `W` and `b` is executed when the loss function `() -> sum(linear(x))` is called, and hence this dependence is visible to Zygote:

```@docs
gradient
```

```julia
julia> W = rand(2, 5); b = rand(2);

julia> linear(x) = W * x .+ b
linear (generic function with 2 methods)

julia> grads = gradient(() -> sum(linear(x)), Params([W, b]))
Grads(...)

julia> grads[W], grads[b] # access gradients using arrays as keys
([0.652543 … 0.683588], [1.0, 1.0])
```

Here `grads` is a dictionary-like object, whose keys are the same parameters we indicated in `Params`. (In fact it wraps a dictionary using `objectid(W)` as keys, which does not change if the values in `W` are mutated).

This implicit style is the one presently used by [Flux.jl](https://github.com/FluxML/Flux.jl), a closely related machine learning library. It uses structs like `Linear` above to define layers, and the function `Flux.params(model)` returns a `Params` object containing all the parameters of all layers. See [its documentation](https://fluxml.ai/Flux.jl/stable/models/basics/) for more details. When using Zygote for most other purposes, however, the explicit style is usually preferred.
