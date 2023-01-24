using Zygote, Test, LinearAlgebra

@testset "basic" begin

@test gradient(x -> real(abs(x)*exp(im*angle(x))), 10+20im)[1] ≈ 1
@test gradient(x -> imag(real(x)+0.3im), 0.3)[1] ≈ 0
@test gradient(x -> imag(conj(x)+0.3im), 0.3 + 0im)[1] ≈ -1im
@test gradient(x -> imag(conj(x)+0.3im), 0.3)[1] ≈ 0  # projected to zero
@test gradient(x -> abs((imag(x)+0.3)), 0.3 + 0im)[1] ≈ 1im
@test gradient(x -> abs((imag(x)+0.3)), 0.3)[1] ≈ 0

@test gradient(a -> real((a*conj(a))), 0.3im)[1] == 0.6im
@test gradient(a -> real((a.*conj(a))), 0.3im)[1] == 0.6im
@test gradient(a -> real(([a].*conj([a])))[], 0.3im)[1] == 0.6im
@test gradient(a -> real(([a].*conj.([a])))[], 0.3im)[1] == 0.6im
@test gradient(a -> real.(([a].*conj.([a])))[], 0.3im)[1] == 0.6im

@test gradient(x -> norm((im*x) ./ (im)), 2)[1] == 1
@test gradient(x -> norm((im) ./ (im*x)), 2)[1] == -1/4
@test gradient(x -> real(det(x)), [1 2im; 3im 4])[1] ≈ [4 3im; 2im 1]
@test gradient(x -> real(logdet(x)), [1 2im; 3im 4])[1] ≈ [4 3im; 2im 1]/10
@test gradient(x -> real(logabsdet(x)[1]), [1 2im; 3im 4])[1] ≈ [4 3im; 2im 1]/10

# https://github.com/FluxML/Zygote.jl/issues/705
@test gradient(x -> imag(sum(exp, x)), [1,2,3])[1] ≈ real(im .* exp.(1:3))
@test gradient(x -> imag(sum(exp, x)), [1+0im,2,3])[1] ≈ im .* exp.(1:3)

end # @testset

fs_C_to_R = (real,
             imag,
             abs,
             abs2,
             z -> abs(z)*cos(im*angle(z)),
             z->abs(cos(exp(z))),
             z->3*real(z)^3-2*imag(z)^5
             )
@testset "C->R" begin
    for f in fs_C_to_R
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote = gradient(real∘f, z)[1]
            ε = 1e-8
            grad_fd = (f(z+ε)-f(z))/ε + im*(f(z+ε*im)-f(z))/ε
            @test abs(grad_zygote - grad_fd) < sqrt(ε)
        end
    end
end

fs_C_to_C_holomorphic = (cos,
                         exp,
                         log,
                         z->z^2,
                         z->(real(z)+im*imag(z))^2,
                         z->real(z)^2 - imag(z)^2 +2im*(real(z)*imag(z)),
                         z->exp(cos(log(z))),
                         z->abs(z)*exp(im*angle(z)),
                         )
@testset "C->C holomorphic" begin
    for f in fs_C_to_C_holomorphic
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote_r = gradient(real∘f, z)[1]
            grad_zygote_i = gradient(imag∘f, z)[1]
            ε = 1e-8
            grad_fd_r = (f(z+ε)-f(z))/ε
            grad_fd_i = (f(z + ε * im) - f(z)) / (ε * im)
            # Check the function is indeed holomorphic
            @assert abs(grad_fd_r - grad_fd_i) < sqrt(ε)
            # Check Zygote derivatives agree with holomorphic definition
            @test grad_zygote_r ≈ -im*grad_zygote_i
            # Check derivative agrees with finite differences
            @test abs(grad_zygote_r - conj(grad_fd_r)) < sqrt(ε)
        end
    end
end


fs_C_to_C_non_holomorphic = (conj,
                             z->abs(z)+0im,
                             z->im*abs(z),
                             z->abs2(z)+0im,
                             z->im*abs2(z),
                             z->z'z,
                             z->conj(z)*z^2,
                             z->imag(z)^2+real(sin(z))^3*1im,
                             )
@testset "C->C non-holomorphic" begin
    for f in fs_C_to_C_non_holomorphic
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote_r = gradient(real∘f, z)[1]
            grad_zygote_i = gradient(imag∘f, z)[1]
            ε = 1e-8
            grad_fd_r = real(f(z+ε)-f(z))/ε + im*real(f(z+ε*im)-f(z))/ε
            grad_fd_i = imag(f(z+ε)-f(z))/ε + im*imag(f(z+ε*im)-f(z))/ε
            # Check derivative of both real and imaginary parts of f as these may differ
            # for non-holomorphic functions
            @test abs(grad_zygote_r - grad_fd_r) < sqrt(ε)
            @test abs(grad_zygote_i - grad_fd_i) < sqrt(ε)
        end
    end
end

@testset "issue 342" begin
    @test Zygote.gradient(x->real(x + 2.0*im), 3.0) == (1.0,)
    @test Zygote.gradient(x->imag(x + 2.0*im), 3.0) == (0.0,)
end

@testset "issue 402" begin
    A = [1,2,3.0]
    y, B_getindex = Zygote.pullback(x->getindex(x,2,1),Diagonal(A))
    bA = B_getindex(1)[1]
    @test bA isa Diagonal
    @test bA == [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
end

@testset "issue #917" begin
    function fun(v)
        c = v[1:3] + v[4:6]*im
        r = v[7:9]
        sum(r .* abs2.(c)) # This would be calling my actual function depending on r and c
    end
    @test Zygote.hessian(fun, collect(1:9)) ≈ [14 0 0 0 0 0 2 0 0; 0 16 0 0 0 0 0 4 0; 0 0 18 0 0 0 0 0 6; 0 0 0 14 0 0 8 0 0; 0 0 0 0 16 0 0 10 0; 0 0 0 0 0 18 0 0 12; 2 0 0 8 0 0 0 0 0; 0 4 0 0 10 0 0 0 0; 0 0 6 0 0 12 0 0 0]
end
