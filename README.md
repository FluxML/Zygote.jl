<p align="center">
<img width="400px" src="https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/zygote.png"/>
</p>

[![Build Status](https://travis-ci.org/FluxML/Zygote.jl.svg?branch=master)](https://travis-ci.org/FluxML/Zygote.jl) [![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.ai/Zygote.jl/dev)

`] add Zygote`

Zygote provides source-to-source automatic differentiation (AD) in Julia, and is the next-gen AD system for the [Flux](https://github.com/FluxML/Flux.jl) differentiable programming framework. For more details and benchmarks of Zygote's technique, see [our paper](https://arxiv.org/abs/1810.07951). You may want to check out Flux for more interesting examples of Zygote usage; the documentation here focuses on internals and advanced AD usage.

Zygote supports Julia 1.0 onwards, but we highly recommend using Julia 1.3 or later.

```julia
julia> using Zygote

julia> f(x) = 5x + 3

julia> f(10), f'(10)
(53, 5)

julia> @code_llvm f'(10)
define i64 @"julia_#625_38792"(i64) {
top:
  ret i64 5
}
```

An example to differentiate a general function gg: R2->R3.

```julia
julia> g1(x)=cos(x[1])+sin(x[2])
g1 (generic function with 1 method)

julia> g2(x)=x[1]^2 + x[2]^2
g2 (generic function with 1 method)

julia> g3(x)=log(x[1]) + exp(x[2])
g3 (generic function with 1 method)

julia> gg(x)= [g1(x);g2(x);g3(x)]
gg (generic function with 1 method)

julia> x=[1.;2.]
2-element Array{Float64,1}:
 1.0
 2.0

julia> Zygote.gradient(g1,x)
([-0.8414709848078965, -0.4161468365471424],)

julia> Zygote.gradient(g2,x)
([2.0, 4.0],)

julia> Zygote.gradient(g3,x)
([1.0, 7.38905609893065],)

julia> Zygote.forward_jacobian(gg,x)[2]
2×3 Array{Float64,2}:
 -0.841471  2.0  1.0
 -0.416147  4.0  7.38906

julia> Zygote.hessian(g1,x)
2×2 Array{Float64,2}:
 -0.540302   0.0
  0.0       -0.909297
```


"Source-to-source" means that Zygote hooks into Julia's compiler, and generates the backwards pass for you – as if you had written it by hand.

Without compromising on performance, Zygote supports the full flexibility and dynamism of the Julia language, including control flow, recursion, closures, structs, dictionaries, and more.

```julia
julia> fs = Dict("sin" => sin, "cos" => cos, "tan" => tan);

julia> gradient(x -> fs[readline()](x), 1)
sin
0.5403023058681398
```

Defining custom gradients is a cinch, and errors have good stacktraces.

```julia
julia> using Zygote: @adjoint

julia> add(a, b) = a + b

julia> @adjoint add(a, b) = add(a, b), Δ -> (Δ, Δ)
```

To support large machine learning models with many parameters, Zygote can differentiate implicitly-used parameters, as opposed to just function arguments.

```julia
julia> W, b = rand(2, 3), rand(2);

julia> predict(x) = W*x .+ b;

julia> g = gradient(Params([W, b])) do
         sum(predict([1,2,3]))
       end
Grads(...)

julia> g[W], g[b]
([1.0 2.0 3.0; 1.0 2.0 3.0], [1.0, 1.0])
```
