
using FFTW
using FillArrays
# FFTW functions do not work with FillArrays, which are needed
# for some functionality of Zygote.. To make it work with FillArrays
# as well, overload the functions
FFTW.fft(x::Fill) = FFTW.fft(collect(x))
FFTW.ifft(x::Fill) = FFTW.ifft(collect(x))
FFTW.fft(x::Fill, dims) = FFTW.fft(collect(x), dims)
FFTW.ifft(x::Fill, dims) = FFTW.ifft(collect(x), dims)


# the adjoint jacobian of an FFT with respect to its input is the reverse FFT of the
# gradient of its inputs.
@adjoint function FFTW.fft(xs)
    return FFTW.fft(xs), function(Δ)
        N = length(xs)
        return (N * FFTW.ifft(Δ),)
    end
end

@adjoint function FFTW.ifft(xs)
    return FFTW.ifft(xs), function(Δ)
        N = length(xs)
        return (1/N* FFTW.fft(Δ),)
    end
end

@adjoint function FFTW.fft(xs, dims)
    # up to now only works when dims is a single integer
    return FFTW.fft(xs, dims), function(Δ)
        # dims can be int, array or tuple,
        # if it is not a single int, convert to array so that we can use it
        # for indexing
        if typeof(dims) != Int
            dims = collect(dims)
        end
        # we need to multiply by all dimensions that we FFT over
        N = prod(size(xs)[dims])
        return (N * FFTW.ifft(Δ, dims), nothing)
    end
end

@adjoint function FFTW.ifft(xs,dims)
    # up to now only works when dims is a single integer
    return FFTW.ifft(xs, dims), function(Δ)
        # dims can be int, array or tuple,
        # if it is not a single int, convert to array so that we can use it
        # for indexing
        if typeof(dims) != Int
            dims = collect(dims)
            end
        # we need to divide by all dimensions that we FFT over
        N = prod(size(xs)[dims])
        return (1/N * FFTW.fft(Δ, dims),nothing)
    end
end
