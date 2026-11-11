# Fused Nested Broadcasts
# =======================

# `@. a * tanh(b) + c` lowers to `materialize(broadcasted(+, broadcasted(*, a, broadcasted(tanh, b)), c))`.
# Differentiated call by call, each node is materialised, with its own kernels in both passes, which
# makes elementwise code on the GPU launch- and bandwidth-bound (#1695). `instrument_broadcasts!`
# rewrites such trees into `zbroadcasted` and `zmaterialize`. With GPU arrays, the former build the tree
# lazily, as a `ZBroadcasted`, and the latter differentiates it as one broadcast of `Dual`s. Nodes with
# rules of their own (ChainRules' `rrule`s, such as NNlib's for activations, or `@adjoint`s) and nodes
# `Dual`s can't differentiate are materialised first, by the usual `broadcasted` rules. Without GPU
# arrays, `zbroadcasted` is `broadcasted`.

zbroadcasted(f, args...) = broadcasted(f, args...)
zmaterialize(x, spec) = materialize(x)

struct ZBroadcasted{F,A<:Tuple}
  f::F
  args::A
end

# A node of a tree already materialised, with its pullback.
struct ZMaterialized{T,B}
  val::T
  back::B
end

_zlazy(x) = x isa Union{AnyGPUArray, ZBroadcasted}

function _pullback(cx::AContext, ::typeof(zbroadcasted), f::F, args::Vararg{Any,N}) where {F,N}
  (any(_zlazy, args) && !isderiving()) || return _pullback(cx, broadcasted, f, args...)
  return ZBroadcasted(f, args), zbroadcasted_back
end
zbroadcasted_back(::Nothing) = nothing
zbroadcasted_back(Δ::NamedTuple{(:f, :args)}) = (nothing, Δ.f, Δ.args...)

function _pullback(cx::AContext, ::typeof(zmaterialize), x, spec)
  y, back = _pullback(cx, materialize, x)
  zmaterialize_back(Δ) = (d = back(Δ); d === nothing ? nothing : (nothing, d[2], nothing))
  return y, zmaterialize_back
end

function _pullback(cx::AContext, ::typeof(zmaterialize), bc::ZBroadcasted, spec::Val)
  y, back = _zmaterialize(cx, bc, spec)
  zmaterialize_fused_back(Δ) = (nothing, back(Δ), nothing)
  return y, zmaterialize_fused_back
end

# The value of `bc`, and a pullback returning its cotangent `(f = df, args = dargs)`.
function _zmaterialize(cx, bc::ZBroadcasted, spec)
  _zfusable(cx, bc) || return _zeager(cx, bc, spec)
  bc′ = _zprep(cx, bc, spec)
  _znodes(bc′) > 1 || return _zeager(cx, bc′, spec)
  fused, argpos, leafmap = _zplan(bc′, spec)
  args = _zpick(_zleaves(bc′), argpos)
  T = Broadcast.combine_eltypes(dual_function(fused), args)
  T == Bool && return broadcast(fused, args...), _ -> nothing
  # The leaves include materialised nodes, which may be complex.
  isconcretetype(T) && T <: Dual && all(a -> eltype(a) <: Real, args) || return _zeager(cx, bc′, spec)
  return _zfuse(bc′, fused, args, leafmap)
end

# By its own `broadcasted` rule, with its lazy arguments materialised.
function _zeager(cx, bc::ZBroadcasted, spec)
  cs = map((a, s) -> _zeager_arg(cx, a, s), bc.args, _zchildspecs(spec, bc))
  y, back = _pullback(cx, broadcasted, bc.f, map(first, cs)...)
  backs = map(last, cs)
  function zeager_back(Δ)
    Δ === nothing && return nothing
    d = back(Δ)
    d === nothing && return nothing
    return (f = d[2], args = map(_zargback, backs, Base.tail(Base.tail(d))))
  end
  return y, zeager_back
end

_zeager_arg(cx, a, s) = (a, nothing)
_zeager_arg(cx, a::ZBroadcasted, s) = _zmaterialize(cx, a, s)
_zeager_arg(cx, a::ZMaterialized, s) = (a.val, a.back)

_zargback(back, d) = back(d)
_zargback(::Nothing, d) = d
_zargback(back, ::Nothing) = nothing
_zargback(::Nothing, ::Nothing) = nothing

# Materialise the nodes that can't be fused into their parents.
_zprep(cx, bc::ZBroadcasted, spec) =
  ZBroadcasted(bc.f, map((a, s) -> _zprep_arg(cx, a, s), bc.args, _zchildspecs(spec, bc)))
_zprep_arg(cx, a, s) = a
_zprep_arg(cx, a::ZBroadcasted, s) =
  _zfusable(cx, a) ? _zprep(cx, a, s) : ZMaterialized(_zmaterialize(cx, a, s)...)

_znodes(T::Type) = T <: ZBroadcasted ? 1 + sum(_znodes, fieldtypes(T.parameters[2]); init = 0) : 0
@generated _znodes(bc::ZBroadcasted) = _znodes(bc)

_zisnode(s, n) = s isa Tuple && length(s) == n + 1 && first(s) === :node

# The specs of `bc`'s arguments, or `nothing`s when `spec` doesn't describe it, as for an argument
# computed by a `zbroadcasted` without GPU arrays.
@generated function _zchildspecs(::Val{S}, bc::ZBroadcasted) where {S}
  n = fieldcount(fieldtype(bc, :args))
  return Expr(:tuple, map(Val, _zisnode(S, n) ? S[2:end] : ntuple(_ -> nothing, n))...)
end

# Which Nodes Fuse
# ----------------

# Zygote's rules for these broadcasts agree with `Dual`s.
const ZDualRules = Union{typeof.((+, -, *, /, Base.literal_pow, identity, tanh, exp, log, sqrt, inv,
                                  conj, real, imag, abs2))...}

_zleaf(T::Type) = T <: ZBroadcasted || Base.issingletontype(T) || T <: Union{AbstractFloat, Integer} ||
  T <: AnyGPUArray && eltype(T) <: Union{AbstractFloat, Integer}

# The type of the materialised `T`, as far as rules dispatch on it: that of its widest array.
function _zstandin(T::Type)
  T <: ZBroadcasted || return T
  ts = filter(t -> t <: AbstractArray, map(_zstandin, fieldtypes(T.parameters[2])))
  return isempty(ts) ? Any : ts[argmax(map(ndims, ts))]
end

# Whether a node `BC`, in context `C`, is a pure function of plain leaves that has no rule of its own,
# but for those of `ZDualRules`. Then also the method instances that a later `@adjoint` or `rrule`
# would displace, as edges of `_zfusable(cx, bc)` (defined in interface2.jl, so that `refresh` redoes it).
function _zfusable(C::Type, BC::Type, world::UInt)
  F, A = BC.parameters
  _dual_purefun(F) && all(_zleaf, fieldtypes(A)) || return false, nothing
  As = map(_zstandin, fieldtypes(A))
  sigs = (Tuple{typeof(_pullback), C, typeof(broadcasted), F, As...},
          Tuple{typeof(rrule), ZygoteRuleConfig{C}, typeof(broadcasted), F, As...},
          Tuple{typeof(rrule), typeof(broadcasted), F, As...})
  matches = map(sig -> Base._which(sig; world, raise = false), sigs)
  any(isnothing, matches) && return false, nothing
  m = matches[1].method
  ok = if m.sig == Tuple{typeof(_pullback), AContext, Any, Vararg{Any}}  # no `@adjoint`
    !first(has_chain_rrule(Tuple{ZygoteRuleConfig{C}, typeof(broadcasted), F, As...}, world))
  else
    m.module === Zygote && F <: ZDualRules
  end
  return ok, ok ? map(Core.Compiler.specialize_method, matches) : nothing
end

_zfusable_true(cx, bc) = true
_zfusable_false(cx, bc) = false

function _zfusable_generator(world::UInt, source, self, cx, bc)
  ok, edges = _zfusable(cx, bc, world)
  template = ok ? _zfusable_true : _zfusable_false
  ci = copy(Base.uncompressed_ast(Base._which(Tuple{typeof(template), Any, Any}; world).method))
  edges === nothing && return ci
  ci.edges = ci.edges isa Core.SimpleVector ? Core.svec(ci.edges..., edges...) :
    Any[something(ci.edges, Any[])..., edges...]
  return ci
end

# The Fused Function
# ------------------

# The function of a fused tree `T`, of `ZNode`s calling `F.instance` on their children, and leaves:
# the `i`-th argument, or a constant `v`.
struct ZFused{T} end
struct ZNode{F,A<:Tuple} end
struct ZArg{i} end
struct ZConst{v} end

function _zleaves!(out, T, s)
  T <: ZBroadcasted || return push!(out, T => s)
  Cs = fieldtypes(T.parameters[2])
  for (i, C) in enumerate(Cs)
    _zleaves!(out, C, _zisnode(s, length(Cs)) ? s[i+1] : nothing)
  end
  return out
end

function _zconst(T, s)
  Base.issingletontype(T) && return ZConst{T.instance}
  s isa Tuple && first(s) === :lit && typeof(s[2]) === T && return ZConst{s[2]}
  return nothing
end

# The fused function of `bc`, the leaves it takes as arguments, and for each leaf, the argument whose
# cotangent it gets (or `0`). A literal or singleton leaf is a constant, and the leaves that are the same
# variable one argument, so that each has at most one lane in the `Dual`s.
@generated function _zplan(bc::ZBroadcasted, ::Val{S}) where {S}
  argpos, leafmap, kinds = Int[], Int[], Any[]
  vars = Dict{Int,Int}()
  for (k, (T, s)) in enumerate(_zleaves!(Pair{Any,Any}[], bc, S))
    c = _zconst(T, s)
    var = s isa Tuple && first(s) === :var ? s[2] : nothing
    if c !== nothing
      push!(kinds, c); push!(leafmap, 0)
    elseif var !== nothing && haskey(vars, var)
      push!(kinds, ZArg{vars[var]}); push!(leafmap, 0)
    else
      push!(argpos, k)
      var === nothing || (vars[var] = length(argpos))
      push!(kinds, ZArg{length(argpos)}); push!(leafmap, length(argpos))
    end
  end
  k = Ref(0)
  tree(T) = T <: ZBroadcasted ?
    ZNode{T.parameters[1], Tuple{map(tree, fieldtypes(T.parameters[2]))...}} : kinds[k[] += 1]
  return :(ZFused{$(tree(bc))}(), Val($(Tuple(argpos))), Val($(Tuple(leafmap))))
end

_zemit!(body, ::Type{ZArg{i}}) where {i} = :(u[$i])
_zemit!(body, ::Type{ZConst{v}}) where {v} = QuoteNode(v)
function _zemit!(body, ::Type{ZNode{F,A}}) where {F,A}
  xs = map(C -> _zemit!(body, C), fieldtypes(A))
  t = gensym(:z)
  push!(body, :($t = _zcall($(QuoteNode(F.instance)), $(xs...))))
  return t
end

@generated function (::ZFused{T})(u::Vararg{Any,N}) where {T,N}
  body = Any[]
  r = _zemit!(body, T)
  return Expr(:block, Expr(:meta, :inline), body..., r)
end

# Predicates see plain values, as in the `Bool` path of the unfused broadcast: ForwardDiff breaks ties
# between `Dual`s using their partials (#1597).
@inline function _zcall(f::F, args::Vararg{Any,N}) where {F,N}
  y = f(args...)
  y isa Bool && return f(map(_zvalue, args)...)
  return _znanguard(y, args...)
end
@inline _zcall(::typeof(abs), x::Dual) = _abs_dualsafe(x)
# A constant branch, as in `ifelse.(x .> 0, x, 0f0)`, would make the element type a `Union`.
@inline _zcall(::typeof(ifelse), c::Bool, x, y) = ifelse(c, promote(x, y)...)

_zvalue(x) = x
_zvalue(x::Dual) = value(x)

# A node whose inputs have zero partials in a lane has a zero partial there, even where its own
# derivative is infinite (only then can the partial differ, so only non-finite ones are checked).
# Otherwise `sqrt.(x.^2)` at `x == 0` would give `0 * Inf == NaN`, which the unfused reverse pass
# guards against in `_pow_grad` (#1598).
@inline _znanguard(y, args...) = y
@inline function _znanguard(y::Dual{T,V,N}, args::Vararg{Any,M}) where {T,V,N,M}
  ps = ntuple(Val(N)) do j
    p = partials(y, j)
    isfinite(p) || _zhaslane(j, args...) ? p : zero(p)
  end
  return Dual{T}(value(y), Partials(ps))
end
@inline _zhaslane(j) = false
@inline _zhaslane(j, x::Dual, xs...) = !iszero(partials(x, j)) || _zhaslane(j, xs...)
@inline _zhaslane(j, x, xs...) = _zhaslane(j, xs...)

# Fusion
# ------

_zleaves(x) = (x,)
_zleaves(x::ZMaterialized) = (x.val,)
_zleaves(bc::ZBroadcasted) = _zleaves_args(bc.args...)
_zleaves_args() = ()
_zleaves_args(a, as...) = (_zleaves(a)..., _zleaves_args(as...)...)

@generated _zpick(xs::Tuple, ::Val{P}) where {P} = Expr(:tuple, [:(xs[$i]) for i in P]...)
@generated _zspread(ds::Tuple, ::Val{M}) where {M} = Expr(:tuple, [j == 0 ? nothing : :(ds[$j]) for j in M]...)

# The cotangent of the tree `bc` from those of its leaves `ds`.
_zrenest(bc, ds) = first(_zrenest1(bc, ds))
_zrenest1(x, ds) = (first(ds), Base.tail(ds))
_zrenest1(x::ZMaterialized, ds) = (_zargback(x.back, first(ds)), Base.tail(ds))
function _zrenest1(bc::ZBroadcasted, ds)
  dargs, rest = _zrenest_args(bc.args, ds)
  return (f = nothing, args = dargs), rest
end
_zrenest_args(::Tuple{}, ds) = (), ds
function _zrenest_args(args::Tuple, ds)
  d1, r1 = _zrenest1(first(args), ds)
  dt, r2 = _zrenest_args(Base.tail(args), r1)
  return (d1, dt...), r2
end

function _zfuse(bc::ZBroadcasted, fused, args, leafmap)
  y, back = _broadcast_forward_recompute(fused, args...)
  bback = broadcast_back(back, args...)
  function zfused_back(Δ)
    Δ === nothing && return nothing
    d = bback(Δ)
    d === nothing && return nothing
    return _zrenest(bc, _zspread(Base.tail(Base.tail(d)), leafmap))
  end
  return y, zfused_back
end
