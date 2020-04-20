using SnoopCompile
println("loading infer benchmark")

@snoopi_bench "Zygote" using Zygote
