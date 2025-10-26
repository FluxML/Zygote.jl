@testitem "pythoncall" skip=(VERSION < v"1.11") begin

  using PythonCall: pyimport, pyconvert

  @testset "PythonCall custom @adjoint" begin
    math = pyimport("math")
    pysin(x) = math.sin(x)
    Zygote.@adjoint pysin(x) = pyconvert(Float64, math.sin(x)), δ -> (pyconvert(Float64, δ * math.cos(x)),)
    @test Zygote.gradient(pysin, 1.5) == Zygote.gradient(sin, 1.5)
  end
end
