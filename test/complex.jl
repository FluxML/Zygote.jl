using Zygote, Test

@test gradient(x -> real(abs(x)*exp(im*angle(x))), 10+20im)[1] ≈ 1
@test gradient(x -> imag(real(x)+0.3im), 0.3)[1] ≈ 0
@test gradient(x -> imag(conj(x)+0.3im), 0.3)[1] ≈ -1im
@test gradient(x -> abs((imag(x)+0.3)), 0.3)[1] == 1im

h1(a) = real((a*conj(a)))
h2(a) = real((a.*conj(a)))
h3(a) = real(([a].*conj([a])))[]
h4(a) = real(([a].*conj.([a])))[]
h5(a) = real.(([a].*conj.([a])))[]

@testset "broadcast" begin
@test h1'(0.3im) == 0.6im
@test h2'(0.3im) == 0.6im
@test h3'(0.3im) == 0.6im
@test h4'(0.3im) == 0.6im
@test h5'(0.3im) == 0.6im
end
