using SnoopCompile

@snoopi_bot "Zygote" begin
  using Zygote
  include(joinpath(pkgdir(Zygote),"src","precompile.jl"))
end
