<p align="center">
  <img width="400px" src="https://raw.githubusercontent.com/FluxML/Zygote.jl/master/docs/src/assets/logo.png#gh-light-mode-only"/>
  <img width="400px" src="https://raw.githubusercontent.com/FluxML/Zygote.jl/master/docs/src/assets/logo-dark.png#gh-dark-mode-only"/>
</p>

<!-- [![Build Status](https://travis-ci.org/FluxML/Zygote.jl.svg?branch=master)](https://travis-ci.org/FluxML/Zygote.jl) -->
[![CI Testing](https://github.com/FluxML/Zygote.jl/workflows/CI/badge.svg)](https://github.com/FluxML/Zygote.jl/actions)
[![Coverage](https://codecov.io/gh/FluxML/Zygote.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Zygote.jl) 
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.ai/Zygote.jl/dev)

`] add Zygote`

Zygote provides source-to-source automatic differentiation (AD) in Julia, and is the next-gen AD system for the [Flux](https://github.com/FluxML/Flux.jl) differentiable programming framework. For more details and benchmarks of Zygote's technique, see [our paper](https://arxiv.org/abs/1810.07951). You may want to check out Flux for more interesting examples of Zygote usage; the documentation here focuses on internals and advanced AD usage.

Zygote supports Julia 1.10 onwards.

```julia
julia> using Zygote

julia> f(x) = 5x + 3

julia> f(10), f'(10)
(53, 5.0)

julia> @code_llvm f'(10)
define i64 @"julia_#625_38792"(i64) {
top:
  ret i64 5
}
```

"Source-to-source" means that Zygote hooks into Julia's compiler, and generates the backwards pass for you – as if you had written it by hand.

Zygote supports the flexibility and dynamism of the Julia language, including control flow, recursion, closures, structs, dictionaries, and more.
Mutation and exception handling are currently not supported.

```julia
julia> fs = Dict("sin" => sin, "cos" => cos, "tan" => tan);

julia> gradient(x -> fs[readline()](x), 1)
sin
0.5403023058681398
```

Zygote benefits from using the [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) ruleset.
Custom gradients can be defined by extending the [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)'s `rrule`:

```julia
julia> using ChainRulesCore

julia> add(a, b) = a + b

julia> function ChainRulesCore.rrule(::typeof(add), a, b)
           add_pb(dy) = (NoTangent(), dy, dy)
           return add(a, b), add_pb
       end
```

To support large machine learning models with many parameters, Zygote can differentiate
whole models with respect to their (possibly nested) structure of parameters, by passing
them explicitly as arguments.

```julia
julia> using Zygote

julia> model = (W = rand(2, 3), b = rand(2));

julia> predict(model, x) = model.W * x .+ model.b;

julia> g = gradient(m -> sum(predict(m, [1, 2, 3])), model)[1]
(W = [1.0 2.0 3.0; 1.0 2.0 3.0], b = [1.0, 1.0])
```

> [!WARNING]
> Zygote also has a legacy *implicit-parameters* interface, in which the parameters of
> interest are collected in a `Zygote.Params` object and the gradients returned in a
> dictionary-like `Grads` object. This interface is **deprecated** and will be removed in a
> future release; use the explicit style shown above instead.
