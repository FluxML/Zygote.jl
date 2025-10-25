@testitem "gradcheck pt. 2" setup=[GradCheckSetup] begin

using Random
using LinearAlgebra
using Statistics
using SparseArrays
using FillArrays
using AbstractFFTs
using FFTW
using Distances
using Zygote: gradient, Buffer
using Base.Broadcast: broadcast_shape
using Distributed: pmap, CachingPool, workers

import FiniteDifferences
import LogExpFunctions

@testset "backsolve" begin
  rng, M, P, Q = MersenneTwister(123456), 13, 10, 9
  X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)
  A, B = randn(rng, P, M), randn(P, Q)
  D = collect(Diagonal(randn(rng, P)))
  L = collect(LowerTriangular(randn(rng, P, P)))
  L[diagind(L)] .= 1 .+ 0.01 .* randn(rng, P)
  U = collect(UpperTriangular(randn(rng, P, P)))
  U[diagind(U)] .= 1 .+ 0.01 .* randn(rng, P)

  # \ (Dense square)
  @test gradtest(\, X, Y)
  @test gradtest(\, X, y)

  # \ (Dense rectangular)
  @test gradtest(\, A, Y)
  @test gradtest(\, A, y)
  @test gradtest(\, B, Y)
  @test gradtest(\, B, y)

  # \ (Diagonal)
  @test gradtest(\, D, Y)
  @test gradtest(\, D, y)
  @test gradtest((D, Y)-> Diagonal(D) \ Y, D, Y)
  @test gradtest((D, Y)-> Diagonal(D) \ Y, D, y)

  # \ (LowerTriangular)
  @test gradtest(\, L, Y)
  @test gradtest(\, L, y)
  @test gradtest((L, Y) -> LowerTriangular(L) \ Y, L, Y)
  @test gradtest((L, Y) -> LowerTriangular(L) \ Y, L, y)

  # \ (UpperTriangular)
  @test gradtest(\, U, Y)
  @test gradtest(\, U, y)
  @test gradtest((U, Y) -> UpperTriangular(U) \ Y, U, Y)
  @test gradtest((U, Y) -> UpperTriangular(U) \ Y, U, y)

  # /
  @test gradtest(/, Y', X)
  @test gradtest((y, X)->y' / X, y, X)

  # / (rectangular)
  @test gradtest(/, Y', A')
  @test gradtest((y, A)->y' / A', y, A)
  @test gradtest(/, Y', B')
  @test gradtest((y, A)->y' / A', y, B)

  # / (Diagonal)
  @test gradtest((D, Y) -> Y' / D, D, Y)
  @test gradtest((D, Y) -> Y' / D, D, y)
  @test gradtest((D, Y)-> Y' / Diagonal(D), D, Y)
  @test gradtest((D, Y)-> Y' / Diagonal(D), D, y)

  # / (LowerTriangular)
  @test gradtest((L, Y) -> Y' / L, L, Y)
  @test gradtest((L, Y) -> Y' / L, L, y)
  @test gradtest((L, Y) -> Y' / LowerTriangular(L), L, Y)
  @test gradtest((L, Y) -> Y' / LowerTriangular(L), L, y)

  # / (UpperTriangular)
  @test gradtest((U, Y) -> Y' / U, U, Y)
  @test gradtest((U, Y) -> Y' / U, U, y)
  @test gradtest((U, Y) -> Y' / UpperTriangular(U), U, Y)
  @test gradtest((U, Y) -> Y' / UpperTriangular(U), U, y)

  # / (UnitLowerTriangular)
  @test gradtest((L, Y) -> Y' / L, L, Y)
  @test gradtest((L, Y) -> Y' / L, L, y)
  @test gradtest((L, Y) -> Y' / UnitLowerTriangular(L), L, Y)
  @test gradtest((L, Y) -> Y' / UnitLowerTriangular(L), L, y)

  # / (UnitUpperTriangular)
  @test gradtest((U, Y) -> Y' / U, U, Y)
  @test gradtest((U, Y) -> Y' / U, U, y)
  @test gradtest((U, Y) -> Y' / UnitUpperTriangular(U), U, Y)
  @test gradtest((U, Y) -> Y' / UnitUpperTriangular(U), U, y)

  @testset "Cholesky" begin
    # Check that the forwards pass computes the correct thing.
    f(X, Y) = cholesky(X * X' + I) \ Y
    @test Zygote.pullback(X -> f(X, Y), X)[1] ≈ cholesky(X * X' + I) \ Y
    @test gradtest(X -> f(X, Y), X)
    @test gradtest(Y -> f(X, Y), Y)
    @test gradtest(X -> f(X, y), X)
    @test gradtest(y -> f(X, y), y)
    g(X) = cholesky(X * X' + I)
    @test Zygote.pullback(g, X)[2]((factors=LowerTriangular(X),))[1] ≈
      Zygote.pullback(g, X)[2]((factors=Matrix(LowerTriangular(X)),))[1]

    # https://github.com/FluxML/Zygote.jl/issues/932
    @test gradcheck(rand(5, 5), rand(5)) do A, x
        C = cholesky(Symmetric(A' * A + I))
        return sum(C \ x) + logdet(C)
    end
  end
end

@testset "Symmetric" begin
  @testset "real" begin
    rng, P = MersenneTwister(123456), 7
    A = randn(rng, P, P)
    @testset "uplo=$uplo" for uplo in (:U, :L)
      @test gradtest(x->Symmetric(x, uplo), A)
      y, back = Zygote.pullback(Symmetric, A, uplo)
      @test y isa Symmetric

      @testset "back(::Diagonal)" begin
        D̄ = Diagonal(randn(rng, P))
        @test back(Diagonal(D̄))[1] isa Diagonal
        @test back(Diagonal(D̄))[1] ≈ back(Matrix(D̄))[1]
      end

      @testset "back(::$TTri)" for TTri in (LowerTriangular,UpperTriangular)
        D̄ = TTri(randn(rng, P, P))
        @test back(D̄)[1] isa Matrix
        @test back(D̄)[2] === nothing
        @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      end
    end
  end

  @testset "complex" begin
    rng, P = MersenneTwister(123456), 7
    Re = randn(rng, P, P)
    Im = randn(rng, P, P)
    A = complex.(Re, Im)

    @testset "gradcheck dense" begin
      for uplo in (:U, :L)
        @test gradcheck(Re,Im) do a, b
          c = Symmetric(complex.(a, b), uplo)
          d = exp.(c)
          sum(real.(d) + imag.(d))
        end
      end
    end

    @testset "uplo=$uplo" for uplo in (:U, :L)
      y, back = Zygote.pullback(Symmetric, A, uplo)
      @test y isa Symmetric

      @testset "back(::Diagonal)" begin
        D̄ = Diagonal(complex.(randn(rng, P), randn(rng, P)))
        @test back(Diagonal(D̄))[1] isa Diagonal
        @test back(Diagonal(D̄))[1] ≈ back(Matrix(D̄))[1]
      end

      @testset "back(::$TTri)" for TTri in (LowerTriangular,UpperTriangular)
        D̄ = TTri(complex.(randn(rng, P, P), randn(rng, P, P)))
        @test back(D̄)[1] isa Matrix
        @test back(D̄)[2] === nothing
        @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      end
    end
  end
end

@testset "Hermitian" begin
  rng, P = MersenneTwister(123456), 7
  Re = randn(rng, P, P)
  Im = randn(rng, P, P)
  A = complex.(Re, Im)

  @testset "gradcheck dense" begin
    for uplo in (:U, :L)
      @test gradcheck(Re,Im) do a, b
        c = Hermitian(complex.(a, b), uplo)
        d = exp.(c)
        sum(real.(d) + imag.(d))
      end
    end
  end

  @testset "uplo=$uplo" for uplo in (:U, :L)
    y, back = Zygote.pullback(Hermitian, A, uplo)
    _, back_sym = Zygote.pullback(Symmetric, A, uplo)
    @test y isa Hermitian

    @testset "back" begin
      D̄ = randn(rng, P, P)
      @test back(D̄)[1] ≈ back_sym(D̄)[1]
    end

    @testset "back(::Diagonal)" begin
      D̄ = Diagonal(complex.(randn(rng, P), randn(rng, P)))
      @test back(D̄)[1] isa Diagonal
      @test back(D̄)[2] === nothing
      @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      @test back(real(D̄))[1] ≈ back_sym(real(D̄))[1]
    end

    @testset "back(::$TTri)" for TTri in (LowerTriangular,UpperTriangular)
      D̄ = TTri(complex.(randn(rng, P, P), randn(rng, P, P)))
      @test back(D̄)[1] isa Matrix
      @test back(D̄)[2] === nothing
      @test back(D̄)[1] ≈ back(Matrix(D̄))[1]
      @test back(real(D̄))[1] ≈ back_sym(real(D̄))[1]
    end
  end
end

@testset "diag" begin
  rng, P = MersenneTwister(123456), 10
  A = randn(rng, P, P)
  @test gradtest(diag, A)
end

@testset "Diagonal" begin
  rng, P = MersenneTwister(123456), 10
  d = randn(rng, P)
  @test gradtest(Diagonal, d)
  y, back = Zygote.pullback(Diagonal, d)
  D̄ = randn(rng, P, P)
  @test back(D̄)[1] ≈ back(Diagonal(D̄))[1]
  @test back(D̄)[1] ≈ back((diag=diag(D̄),))[1]
end

@testset "dense + UniformScaling" begin
  rng = MersenneTwister(123456)
  A, λ = randn(rng, 10, 10), randn(rng)
  @test gradtest(A->A + 5I, A)
  @test gradtest(A->5I - A, A)
  @test gradtest(λ->A + λ[1] * I, [λ])
end

@testset "cholesky" begin
  @testset "cholesky - dense" begin
    rng, N = MersenneTwister(123456), 5
    A = randn(rng, N, N)
    @test cholesky(A' * A + I).U ≈ first(Zygote.pullback(A->cholesky(A' * A + I), A)).U
    @test gradtest(A->cholesky(A' * A + I).U, A)
    @test gradtest(A->logdet(cholesky(A' * A + I)), A)
    @test gradtest(B->cholesky(Symmetric(B)).U, A * A' + I)
    @test gradtest(B->logdet(cholesky(Symmetric(B))), A * A' + I)
  end
  @testset "cholesky - scalar" begin
    rng = MersenneTwister(123456)
    y, back = Zygote.pullback(cholesky, 5.0 * ones(1, 1))
    y′, back′ = Zygote.pullback(cholesky, 5.0)
    C̄ = randn(rng, 1, 1)
    @test back′((factors=C̄,))[1] isa Real
    @test back′((factors=C̄,))[1] ≈ back((factors=C̄,))[1][1, 1]
  end
  @testset "cholesky - Diagonal" begin
    rng, N = MersenneTwister(123456), 3
    D = Diagonal(exp.(randn(rng, N)))
    Dmat = Matrix(D)
    y, back = Zygote.pullback(cholesky, Dmat)
    y′, back′ = Zygote.pullback(cholesky, D)
    C̄ = (factors=randn(rng, N, N),)
    @test back′(C̄)[1] isa Diagonal
    @test diag(back′(C̄)[1]) ≈ diag(back(C̄)[1])
  end
  @testset "cholesky - Hermitian{Complex}" begin
    rng, N = MersenneTwister(123456), 3
    A = randn(rng, Complex{Float64}, N, N)
    H = Hermitian(A * A' + I)
    Hmat = Matrix(H)
    y, back = Zygote.pullback(cholesky, Hmat)
    y′, back′ = Zygote.pullback(cholesky, H)
    C̄ = (factors=randn(rng, N, N),)
    @test only(back′(C̄)) isa Hermitian
    # gradtest does not support complex gradients, even though the pullback exists
    d = only(back(C̄))
    d′ = only(back′(C̄))
    @test (d + d')/2 ≈ d′
  end
  @testset "cholesky - Hermitian{Real}" begin
    rng, N = MersenneTwister(123456), 3
    A = randn(rng, N, N)
    H = Hermitian(A * A' + I)
    Hmat = Matrix(H)
    y, back = Zygote.pullback(cholesky, Hmat)
    y′, back′ = Zygote.pullback(cholesky, H)
    C̄ = (factors=randn(rng, N, N),)
    @test back′(C̄)[1] isa Hermitian
    @test gradtest(B->cholesky(Hermitian(B)).U, Hmat)
    @test gradtest(B->logdet(cholesky(Hermitian(B))), Hmat)
  end
end

@testset "lyap" begin
  rng, N = MersenneTwister(6865943), 5
  for i = 1:5
    A = randn(rng, N, N)
    C = randn(rng, N, N)
    @test gradtest(lyap, A, C)
  end
  @test gradcheck(x->lyap(x[1],x[2]),[3.1,4.6])
end

@testset "matrix exponential" begin
  @testset "real dense" begin
    rng, N = MersenneTwister(6865931), 8
    for i = 1:5
      A = randn(rng, N, N)
      @test gradtest(exp, A)

      @testset "similar eigenvalues" begin
        λ, V = eigen(A)
        λ[1] = λ[3] + sqrt(eps(real(eltype(λ)))) / 10
        A2 = real.(V * Diagonal(λ) / V)
        @test gradtest(exp, A2)
      end
    end
  end

  @testset "complex dense" begin
    rng, N = MersenneTwister(6865931), 8
    for i = 1:5
      A = randn(rng, ComplexF64, N, N)
      @test gradcheck(reim(A)...) do a,b
        c = complex.(a, b)
        d = exp(c)
        return sum(real.(d) + 2 .* imag.(d))
      end

      @testset "similar eigenvalues" begin
        λ, V = eigen(A)
        λ[1] = λ[3] + sqrt(eps(real(eltype(λ)))) / 10
        A2 = V * Diagonal(λ) / V
        @test gradcheck(reim(A2)...) do a,b
          c = complex.(a, b)
          d = exp(c)
          return sum(real.(d) + 2 .* imag.(d))
        end
      end
    end
    A = [ 0.0    1.0    0.0
          0.0    0.0    1.0
          -4.34 -18.31  -0.43]
    _,back = Zygote.pullback(exp,A)
    Ȳ = rand(3,3)
    @test isreal(back(Ȳ)[1])
  end
end

_hermsymtype(::Type{<:Symmetric}) = Symmetric
_hermsymtype(::Type{<:Hermitian}) = Hermitian

function _gradtest_hermsym(f, ST, A; kwargs...)
  gradtest(_splitreim(collect(A))...; kwargs...) do (args...)
    B = f(ST(_joinreim(_dropimaggrad.(args)...)))
    return sum(_splitreim(B))
  end
end

@testset "eigen(::RealHermSymComplexHerm)" begin
  MTs = (Symmetric{Float64}, Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  @testset "eigen(::$MT)" for MT in MTs
    T = eltype(MT)
    ST = _hermsymtype(MT)

    A = ST(randn(rng, T, N, N))
    U = eigvecs(A)

    @test _gradtest_hermsym(ST, A) do (A)
      d, U = eigen(A)
      return U * Diagonal(exp.(d)) * U'
    end

    y = Zygote.pullback(eigen, A)[1]
    y2 = eigen(A)
    @test y.values ≈ y2.values
    @test y.vectors ≈ y2.vectors

    @testset "low rank" begin
      A2 = Symmetric(U * Diagonal([randn(rng), zeros(N-1)...]) * U')
      @test_broken _gradtest_hermsym(ST, A2) do (A)
        d, U = eigen(A)
        return U * Diagonal(exp.(d)) * U'
      end
    end
  end
end

@testset "eigvals(::RealHermSymComplexHerm)" begin
  MTs = (Symmetric{Float64}, Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  @testset "eigvals(::$MT)" for MT in MTs
    T = eltype(MT)
    ST = _hermsymtype(MT)

    A = ST(randn(rng, T, N, N))
    @test _gradtest_hermsym(A ->eigvals(A), ST, A)
    @test Zygote.pullback(eigvals, A)[1] ≈ eigvals(A)
  end
end

@testset "Hermitian/Symmetric power series functions" begin
  MTs = (Symmetric{Float64}, Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  domains = (Real, Complex)
  @testset "$(nameof(f))(::RealHermSymComplexHerm)" for f in (exp, log, cos, sin, tan, cosh, sinh, tanh, acos, asin, atan, acosh, asinh, atanh, sqrt)
    @testset "$(nameof(f))(::$MT)" for MT in MTs
      T = eltype(MT)
      ST = _hermsymtype(MT)
      @testset "domain $domain" for domain in domains
        preA = _randmatseries(rng, f, T, N, domain)
        preA === nothing && continue
        A = ST(preA)
        λ, U = eigen(A)

        @test _gradtest_hermsym(f, ST, A)

        y, back = Zygote.pullback(f, A)
        y2 = f(A)
        @test y ≈ y2
        broken = VERSION >= v"1.12" &&  MT <: Hermitian{Float64} && domain == Real
        @test typeof(y) == typeof(y2) broken=broken
        ȳ = randn(eltype(y), size(y))
        if y isa Union{Symmetric,Hermitian}
            ȳ = typeof(y)(ȳ, y.uplo)
        end
        Ā = back(ȳ)[1]
        @test typeof(Ā) == typeof(A)
        @test Ā.uplo == A.uplo

        @testset "similar eigenvalues" begin
          λ[1] = λ[3] + sqrt(eps(eltype(λ))) / 10
          A2 = U * Diagonal(λ) * U'
          @static if VERSION >= v"1.11" && VERSION < v"1.12"
            broken = f == sqrt && MT <: Symmetric{Float64} && domain == Real
            # @show f MT domain
            @test _gradtest_hermsym(f, ST, A2) broken=broken
          else
            @test _gradtest_hermsym(f, ST, A2)
          end
        end

        if f ∉ (log, sqrt) # only defined for invertible matrices
          @testset "low rank" begin
            A3 = U * Diagonal([rand(rng), zeros(N-1)...]) * U'
            @test _gradtest_hermsym(f, ST, A3)
          end
        end
      end
    end
  end

  @testset "sincos(::RealHermSymComplexHerm)" begin
    @testset "sincos(::$MT)" for MT in MTs
      T = eltype(MT)
      ST = _hermsymtype(MT)
      A = ST(_randmatseries(rng, sincos, T, N, Real))
      λ, U = eigen(A)

      @test gradtest(_splitreim(collect(A))...) do (args...)
        S,C = sincos(ST(_joinreim(_dropimaggrad.(args)...)))
        return vcat(vec.(_splitreim(S))..., vec.(_splitreim(C))...)
      end

      y = Zygote.pullback(sincos, A)[1]
      y2 = sincos(A)
      @test y[1] ≈ y2[1]
      @test typeof(y[1]) == typeof(y2[1])
      @test y[2] ≈ y2[2]
      @test typeof(y[2]) == typeof(y2[2])

      @testset "similar eigenvalues" begin
        λ[1] = λ[3] + sqrt(eps(eltype(λ))) / 10
        A2 = U * Diagonal(λ) * U'
        @test gradtest(_splitreim(collect(A2))...) do (args...)
          S,C = sincos(ST(_joinreim(_dropimaggrad.(args)...)))
          return vcat(vec.(_splitreim(S))..., vec.(_splitreim(C))...)
        end
      end

      @testset "low rank" begin
        A3 = U * Diagonal([rand(rng), zeros(N-1)...]) * U'
        @test gradtest(_splitreim(collect(A3))...) do (args...)
          S,C = sincos(ST(_joinreim(_dropimaggrad.(args)...)))
          return vcat(vec.(_splitreim(S))..., vec.(_splitreim(C))...)
        end
      end
    end
  end

  @testset "^(::RealHermSymComplexHerm, p::Real)" begin
    @testset for p in (-1.0, -0.5, 0.5, 1.0, 1.5)
      @testset "^(::$MT, $p)" for MT in MTs
        T = eltype(MT)
        ST = _hermsymtype(MT)
        @testset "domain $domain" for domain in domains
          A = ST(_randmatseries(rng, ^, T, N, domain))
          λ, U = eigen(A)

          @test gradcheck(_splitreim(collect(A))..., [p]) do (args...)
            p = _dropimaggrad(args[end][1])
            A = ST(_joinreim(_dropimaggrad.(args[1:end-1])...))
            B = A^p
            return abs(sum(sin.(B)))
          end

          y = Zygote.pullback(^, A, p)[1]
          y2 = A^p
          @test y ≈ y2
          @test typeof(y) == typeof(y2)

          @testset "similar eigenvalues" begin
            λ[1] = λ[3] + sqrt(eps(eltype(λ))) / 10
            A2 = U * Diagonal(λ) * U'
            @test gradcheck(_splitreim(collect(A2))..., [p]) do (args...)
              p = _dropimaggrad(args[end][1])
              A = ST(_joinreim(_dropimaggrad.(args[1:end-1])...))
              B = A^p
              return abs(sum(sin.(B)))
            end
          end
        end
      end
    end
  end
end

@testset "^(::Union{Symmetric,Hermitian}, p::Integer)" begin
  MTs = (Symmetric{Float64}, Symmetric{ComplexF64},
         Hermitian{Float64}, Hermitian{ComplexF64})
  rng, N = MersenneTwister(123), 7
  @testset for p in -3:3
    @testset "^(::$MT, $p)" for MT in MTs
      T = eltype(MT)
      ST = _hermsymtype(MT)
      A = ST(randn(rng, T, N, N))

      if p == 0
        @test gradient(_splitreim(collect(A))...) do (args...)
          A = ST(_joinreim(_dropimaggrad.(args)...))
          B = A^p
          return sum(sin.(vcat(vec.(_splitreim(B))...)))
        end === map(_->nothing, _splitreim(A))
      else
        @static if VERSION >= v"1.11"
          # @show MT p
          broken = MT <: Symmetric{Float64} && p == -3
          @test gradtest(_splitreim(collect(A))...) do (args...)
            A = ST(_joinreim(_dropimaggrad.(args)...))
            B = A^p
            return vcat(vec.(_splitreim(B))...)
          end broken=broken
        else
          @test gradtest(_splitreim(collect(A))...) do (args...)
            A = ST(_joinreim(_dropimaggrad.(args)...))
            B = A^p
            return vcat(vec.(_splitreim(B))...)
          end
        end
      end

      y = Zygote.pullback(^, A, p)[1]
      y2 = A^p
      @test y ≈ y2
      @test typeof(y) === typeof(y2)
    end
  end
end

end
