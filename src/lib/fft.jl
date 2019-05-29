using FFTW


# FFTW functions do not work with FillArray. To make it work with FillArrays
# as well, overload the functions
FFTW.ifft(x::Zygote.FillArray) = FFTW.ifft(collect(x))
FFTW.fft(x::Zygote.FillArray) = FFTW.fft(collect(x))
FFTW.ifft(x::Zygote.FillArray, dims) = FFTW.ifft(collect(x), dims)
FFTW.fft(x::Zygote.FillArray, dims) = FFTW.fft(collect(x), dims)

# the gradient of an FFT with respect to its input is the reverse FFT of the
# gradient of its inputs.
Zygote.@adjoint FFTW.fft(xs) = (FFTW.fft(xs), (Δ)-> (FFTW.ifft(Δ),))
Zygote.@adjoint FFTW.ifft(xs) = (FFTW.ifft(xs), (Δ)-> (FFTW.fft(Δ),))
Zygote.@adjoint FFTW.fft(xs,dims) = (FFTW.fft(xs,dims), (Δ)-> (FFTW.ifft(Δ,dims),))
Zygote.@adjoint FFTW.ifft(xs,dims) = (FFTW.ifft(xs,dims), (Δ)-> (FFTW.fft(Δ,dims),))
