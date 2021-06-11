using Pkg: Pkg
Pkg.activate(dirname(@__DIR__))
using Zygote
using BenchmarkTools

const xs = randn(10_000)
@btime Zygote.pullback(sum, abs, $xs)[2](1);

Zygote._pullback(x->2x, 2.2)[2](1.2)


bar(x) = partialsort(x, 1; rev=true)
bar([-3.0, 2, 3])
Zygote._pullback(bar, [-3.0, 2, 3])[2](1)

foo(x) = first(sum(abs, x; dims=1))
foo([-3.0, 2, 3])
Zygote.pullback(foo, [-3.0, 2, 3])[2](1)
