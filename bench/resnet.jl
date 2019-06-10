using Flux
using Flux: expand

struct Conv{F,W,B,D}
  σ::F
  weight::W
  bias::B
  dims::D
end

Flux.DenseConvDims(M, w_size, stride, padding, dilation) =
  DenseConvDims{
      M - 2,
      w_size[1:end-2],
      w_size[end-1],
      w_size[end],
      stride,
      padding,
      dilation,
      false
  }((10, 10))

function Conv(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
              stride = 1, pad = 0, dilation = 0) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  dims = DenseConvDims(ndims(w), size(w), stride, pad, dilation)
  return Conv(σ, w, b, dims)
end

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
     init = randn,  stride = 1, pad = 0, dilation = 1) where N =
  Conv(init(k..., ch...), zeros(ch[2]), σ,
       stride = stride, pad = pad, dilation = dilation)

function (c::Conv)(x::AbstractArray)
  σ = c.σ
  # b = reshape(c.bias, map(_->1, c.stride)..., :, 1)
  # σ.(conv(x, c.weight, cdims) .+ b)
  @which conv(x, c.weight, typeof(c.dims)((size(x, 1), size(x, 2))))
end

Conv((2,2), 3=>4)

Conv((2,2), 3=>4)(randn(100, 100, 3, 2))

x = rand(10, 10, 3, 2)
w = rand(2, 2, 3, 4)
@code_typed conv(x, w, DenseConvDims(x, w))

@treelike Conv

struct ResidualBlock{C,N,S}
  convs::C
  norms::N
  shortcut::S
end

Flux.@treelike ResidualBlock

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
  local convs = []
  local norms = []
  for i in 2:length(filters)
    push!(convs, Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
    push!(norms, BatchNorm(filters[i]))
  end
  ResidualBlock(Tuple(convs),Tuple(norms),shortcut)
end

function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity)
  ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)
end

function (block::ResidualBlock)(input)
  value = input
  for i in 1:length(block.convs)-1
    value = relu.((block.norms[i])((block.convs[i])(value)))
  end
  relu.(((block.norms[end])((block.convs[end])(value))) + block.shortcut(input))
end

function Bottleneck(filters::Int, downsample::Bool = false, res_top::Bool = false)
  if !downsample && !res_top
    return ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
  elseif downsample && res_top
    return ResidualBlock([filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], Chain(Conv((1,1), filters=>4 * filters, pad = (0,0), stride = (1,1)), BatchNorm(4 * filters)))
  else
    shortcut = Chain(Conv((1,1), 2 * filters=>4 * filters, pad = (0,0), stride = (2,2)), BatchNorm(4 * filters))
    return ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
  end
end

function resnet50()
  local layers = [3, 4, 6, 3]
  local layer_arr = []

  push!(layer_arr, Conv((7,7), 3=>64, pad = (3,3), stride = (2,2)))
  push!(layer_arr, MaxPool((3,3), pad = (1,1), stride = (2,2)))

  initial_filters = 64
  for i in 1:length(layers)
    push!(layer_arr, Bottleneck(initial_filters, true, i==1))
    for j in 2:layers[i]
      push!(layer_arr, Bottleneck(initial_filters))
    end
    initial_filters *= 2
  end

  push!(layer_arr, MeanPool((7,7)))
  push!(layer_arr, x -> reshape(x, :, size(x,4)))
  push!(layer_arr, (Dense(2048, 1000)))
  push!(layer_arr, softmax)

  Chain(layer_arr...)
end

m = mapleaves(Flux.data, resnet50())

x = rand(Float32, 200, 200, 3, 5)
@code_typed m.layers[1](x)
