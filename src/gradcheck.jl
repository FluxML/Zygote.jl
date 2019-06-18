export jacobian, njacobian, gradcheck

# Base on torch.gradcheck
make_jacobian(x::AbstractArray{T}, out_length::Int) where T = zeros(T, out_length, length(x))
make_jacobian(x::Number, out_length::Int) = zeros(typeof(x), out_length, 1)

function perturb(f, xs::Tuple, idx::Int, x::AbstractArray, x_idx, eps)
    orig = x[x_idx]
    x[x_idx] = orig + eps
    out = copy(f(xs...))
    x[x_idx] = orig
    return out
end

function perturb(f, xs::Tuple, idx::Int, x::Number, x_idx, eps)
    out = copy(f((k == idx ? x + eps : x for (k, x) in enumerate(xs))...))
    return out
end

zero_like(x::T) where {T <: Number} = zero(T)
zero_like(x) = fill!(similar(x), 0)
zero_like(x::Broadcast.Broadcasted) = zero_like(Broadcast.materialize(x))

_set_njacobian_elem!(d_tensor, d_idx, r::AbstractArray) = d_tensor[:, d_idx] = vec(r)
_set_njacobian_elem!(d_tensor, d_idx, r::Number) = d_tensor[1, d_idx] = r

"""
    njacobian(f, xs...; eps=√eps(eltype(first(xs))))

Return the numerical jacobian of `f` with input `xs...`. It use square root of
the machine precision the first arguement element type `sqrt(eps(eltype(first(xs)))`. 
"""
function njacobian(f, xs...; eps=sqrt(eps(eltype(first(x)))))
    output_size = length(f(xs...))
    jacobians = map(x->make_jacobian(x, output_size), xs)

    for (idx, (x, d_tensor)) in enumerate(zip(xs, jacobians))
        for (d_idx, x_idx) in enumerate(eachindex(x))
            outa = perturb(f, xs, idx, x, x_idx, -eps)
            outb = perturb(f, xs, idx, x, x_idx, eps)
            r = (outb - outa) / 2eps
            _set_njacobian_elem!(d_tensor, d_idx, r)            
        end
    end
    return jacobians
end

"""
    jacobian(f, xs...)

Return the analytical jacobian of `f` with input `xs...`.
"""
function jacobian(f, xs...)
    output, back = forward(f, xs...)
    output_size = length(output)
    jacobians = map(x->make_jacobian(x, output_size), xs)
    grad_output = zero_like(output)
    jacobian!(back, jacobians, grad_output)
    return jacobians
end

# to get numbers through
_vec(x) = x
_vec(x::AbstractArray) = vec(x)

function jacobian!(f_back, jacobians, grad_output::T) where T <: Number    
    grads_input = f_back(one(T))
    for (jacobian_x, d_x) in zip(jacobians, grads_input)
        jacobian_x[1, :] .= _vec(d_x)
    end
    return jacobians
end

function jacobian!(f_back, jacobians, grad_output::AbstractArray)
    for idx in eachindex(grad_output)
        grad_output = fill!(grad_output, 0)
        grad_output[idx] = 1
        grads_input = f_back(grad_output)
        for (jacobian_x, d_x) in zip(jacobians, grads_input)
            jacobian_x[idx, :] .= _vec(d_x)
        end
    end
    return jacobians
end


"""
    gradcheck(f, xs...; eps=sqrt(eps), atol::Real=0, rtol::Real=atol>0 ? 0 : √eps)

Check the gradient of `f` at input `xs...` by comparing numerical jacobian and analytical jacobian.
"""
function gradcheck(f, xs...;
        eps=sqrt(eps(eltype(first(x)))),
        atol::Real=0, rtol::Real= atol > 0 ? 0 : sqrt(eps))
    
    all( isapprox.(njacobian(f, xs...; eps=eps), jacobian(f, xs...); atol=atol, rtol=rtol) )
end
