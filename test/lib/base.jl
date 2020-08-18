struct Maz
    d::Dict
end
maz = m -> begin
    d = getfield(m, :d)
    z = getindex(d, :x)
    return z
end

struct Bar
    x::Float64
end
bar = m -> begin
    z = getfield(m, :x)
    return z
end

fn = m -> begin
    z = getindex(m, :x)
    return z
end

@testset "Grad of closure over dictionary" begin
    _, back = pullback(maz, Maz(Dict(:x => 5.0)))
    @test back(1.0) == ((d = Dict{Any, Any}(:x => 1.0), ), )
    _, back = pullback(bar, Bar(5.0))
    @test back(1.0) == ((x = 1.0, ), )
    _, back = pullback(fn, Dict(:x => 5.0))
    @test back(1.0) == (Dict{Any, Any}(:x => 1.0), )

    @test Zygote.gradient(x -> (() -> x[:y])(), Dict(:y => 0.4)) == (Dict{Any, Any}(:y => 1.0), )
end
