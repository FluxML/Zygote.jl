<p align="center">
<img width="400px" src="https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/zygote.png"/>
</p>

[![Build Status](https://travis-ci.org/FluxML/Zygote.jl.svg?branch=master)](https://travis-ci.org/FluxML/Zygote.jl)

Zygote is a working prototype for source-to-source automatic differentiation (AD) in Julia.

Zygote is in an early stage and may break, but issue reports and beta testing are welcome. Note that due to current compiler limitations it may be faster on [this branch](https://github.com/JuliaLang/julia/tree/mji/zygote) of Julia.

```julia
julia> using Zygote: derivative

julia> f(x) = 3x^2 + 2x + 1
f (generic function with 1 method)

julia> f′(x) = derivative(f, x)
f′ (generic function with 1 method)

julia> f(5), f′(5)
(86, 32.0)
```

"Source-to-source" means that Zygote hooks into Julia's compiler, and generates the backwards pass for you – as if you had written it by hand.

Despite its performance, Zygote supports the full flexibility and dynamism of the Julia language.

```julia
julia> using Zygote: gradient

julia> fs = Dict("sin" => sin, "cos" => cos, "tan" => tan);

julia> gradient(x -> fs[readline()](x), 1)
sin
(0.5403023058681398,)
```

Zygote's lower-level API exposes backpropagators, which can be used to feed custom sensitivities or implement other more advanced functionality.

```julia
julia> using Zygote: forward

julia> y, back = forward(x -> x .* 3, [1, 1, 1])
([3, 3, 3], λ)

julia> back([1, 2, 3])
([3, 6, 9],)
```

Defining custom gradients is a cinch – and if you make a mistake, you'll get a great stack trace pointing you to the issue.

```julia
julia> using Zygote: @grad

julia> add(a, b) = a + b

julia> @grad add(a, b) = add(a, b), Δ -> (Δ, Δ)
```

To support large machine learning models with many parameters, Zygote can differentiate implicitly-used parameters, as opposed to just function arguments.

```julia
julia> W, b = rand(2, 3), rand(2);

julia> predict(x) = W*x .+ b;

julia> g = gradient(() -> sum(predict([1,2,3])), Params([W, b]))
Grads(...)

julia> g[W], g[b]
([1.0 2.0 3.0; 1.0 2.0 3.0], [1.0, 1.0])
```
