using ArgCheck
using OffsetArrays

export tmap, tmap!, tmapreduce, treduce

############################## docs ##############################
function _make_docstring(signature, fname)
    """
        $signature
    Threaded analog of [`Base.$fname`](@ref). See [`Base.$fname`](@ref) for a description of arguments.
    """
end

function tname(s::Symbol)
    Symbol("t"*string(s))
end

const SYMBOLS_MAPREDUCE_LIKE = [:sum, :prod, :minimum, :maximum]
for fun in SYMBOLS_MAPREDUCE_LIKE
    tfun = tname(fun)
    signature = "$tfun([f,] src::AbstractArray)"
    docstring = _make_docstring(signature, fun)
    @eval begin
        export $tfun
        """
        $($docstring)
        """
        function $tfun end
    end
end

"""
$(_make_docstring("tmap!(f, dst::AbstractArray, srcs::AbstractArray...)", :map!))
"""
function tmap! end

"""
$(_make_docstring("tmap(f, srcs::AbstractArray...)", :map))
"""
function tmap end

"""
$(_make_docstring("tmapreduce(f, op, src::AbstractArray [;init])", :map))
"""
function tmapreduce end

"""
$(_make_docstring("treduce(op, src::AbstractArray [;init])", :reduce))
"""
function treduce end

############################## Helper functions ##############################
struct Batches{V}
    firstindex::Int
    lastindex::Int
    batch_size::Int
    values::V
    length::Int
end

function Batches(values, batch_size::Integer)
    @argcheck batch_size > 0
    r = eachindex(values)
    @assert r isa AbstractUnitRange
    len = ceil(Int, length(r) / batch_size)
    Batches(first(r), last(r), batch_size, values, len)
end

Base.length(o::Batches) = o.length
Base.eachindex(o::Batches) = Base.OneTo(length(o))
function Base.getindex(o::Batches, i)
    @boundscheck (@argcheck i in eachindex(o))
    start = o.firstindex + (i-1) * o.batch_size
    stop  = min(start + (o.batch_size) -1, o.lastindex)
    o.values[start:stop]
end

function default_batch_size(len)
    len <= 1 && return 1
    nthreads=Threads.nthreads()
    items_per_thread = len/nthreads
    items_per_batch = items_per_thread/4
    clamp(1, round(Int, items_per_batch), len)
end

function Base.iterate(o::Batches, state=1)
    if state in eachindex(o)
        o[state], state+1
    else
        nothing
    end
end

mutable struct _RollingCutOut{A,I<:AbstractUnitRange,T} <: AbstractVector{T}
    array::A
    eachindex::I
end

function _RollingCutOut(array::AbstractArray, indices)
    T = eltype(array)
    A = typeof(array)
    I = typeof(indices)
    _RollingCutOut{A, I, T}(array, indices)
end

Base.size(r::_RollingCutOut) = (length(r.eachindex),)

function Base.eachindex(r::_RollingCutOut, rs::_RollingCutOut...)
    for r2 in rs
        @assert r.eachindex == r2.eachindex
    end
    Base.IdentityUnitRange(r.eachindex)
end
Base.axes(r::_RollingCutOut) = (eachindex(r),)

@inline function Base.getindex(o::_RollingCutOut, i)
    @boundscheck checkbounds(o, i)
    @inbounds o.array[i]
end

@inline function Base.setindex!(o::_RollingCutOut, val, i)
    @boundscheck checkbounds(o, i)
    @inbounds o.array[i] = val
end

############################## tmap, tmap! ##############################
struct MapWorkspace{F,B,AD,AS}
    f::F
    batches::B
    arena_dst_view::AD
    arena_src_views::AS
end

@noinline function run!(o::MapWorkspace)
    let o=o
        Threads.@threads for i in 1:length(o.batches)
            tid = Threads.threadid()
            dst_view  = o.arena_dst_view[tid]
            src_views = o.arena_src_views[tid]
            inds = o.batches[i]
            dst_view.eachindex = inds
            for view in src_views
                view.eachindex = inds
            end
            Base.map!(o.f, dst_view, src_views...)
        end
    end
end

function create_arena_src_views(srcs, sample_inds)
    nt = Threads.nthreads()
    [Base.map(src -> _RollingCutOut(src, sample_inds), srcs) for _ in 1:nt]
end

@noinline function prepare(::typeof(tmap!), f, dst, srcs; batch_size::Int)
    # we use IndexLinear since _RollingCutOut implementation
    # does not support other indexing well
    all_inds  = eachindex(IndexLinear(), srcs...)
    batches   = Batches(all_inds, batch_size)
    sample_inds = batches[1]
    nt = Threads.nthreads()
    arena_dst_view  = [_RollingCutOut(dst, sample_inds) for _ in 1:nt]
    arena_src_views = create_arena_src_views(srcs, sample_inds)
    return MapWorkspace(f, batches, arena_dst_view, arena_src_views)
end

@noinline function tmap!(f, dst, srcs::AbstractArray...;
                        batch_size=default_batch_size(length(dst)))
    isempty(first(srcs)) && return dst
    w = prepare(tmap!, f, dst, srcs, batch_size=batch_size)
    run!(w)
    dst
end

function tmap(f, srcs::AbstractArray...;
             batch_size=default_batch_size(length(first(srcs)))
            )
    g = Base.Generator(f,srcs...)
    T = Base.@default_eltype(g)
    dst = similar(first(srcs), T)
    tmap!(f, dst, srcs...; batch_size=batch_size)
end

############################## tmapreduce(like) ##############################
struct Reduction{O}
    op::O
end
(red::Reduction)(f, srcs...) = Base.mapreduce(f, red.op, srcs...)

struct MapReduceWorkspace{R,F,B,V,OA<:OffsetArray}
    reduction::R
    f::F
    batches::Batches{B}
    arena_src_views::V
    batch_reductions::OA
end

struct NoInit end

function create_reduction(::typeof(tmapreduce), op)
    Reduction(op)
end

function prepare(::typeof(tmapreduce), f, op, srcs; init, batch_size::Int)
    red = create_reduction(tmapreduce, op)
    w = prepare_mapreduce_like(red, f, srcs, init, batch_size=batch_size)
    return w
end

function tmapreduce(f, op, srcs::AbstractArray...;
                   init=NoInit(),
                   batch_size= default_batch_size(length(first(srcs)))
                  )
    if isempty(first(srcs))
        if init isa NoInit
            return Base.mapreduce(f, op, srcs...)
        else
            return Base.mapreduce(f, op, srcs..., init=init)
        end
    end
    w = prepare(tmapreduce, f, op, srcs, init=init, batch_size=batch_size)
    run!(w)
end

function treduce(op, srcs::AbstractArray...; kw...)
    tmapreduce(identity, op, srcs...; kw...)
end


for red in SYMBOLS_MAPREDUCE_LIKE
    tred = tname(red)
    @eval function $tred end

    @eval function prepare(::typeof($tred), f, srcs; batch_size::Int)
        base_red = Base.$red
        prepare_mapreduce_like(base_red, f, srcs, batch_size=batch_size)
    end

    @eval function $tred(f, src::AbstractArray;
                        batch_size=default_batch_size(length(src)))
        isempty(src) && return Base.$red(f, src)
        srcs = (src,)
        w = prepare($tred, f, srcs, batch_size=batch_size)
        run!(w)
    end
    @eval $tred(src; kw...) = $tred(identity, src; kw...)
end

function prepare_mapreduce_like(red, f, srcs, init=NoInit(); batch_size::Int)
    all_inds  = eachindex(IndexLinear(), srcs...)
    batches   = Batches(all_inds, batch_size)
    sample_inds = batches[1]

    arena_src_views = create_arena_src_views(srcs, sample_inds)
    T = get_return_type(red, f, srcs)

    if (init isa NoInit)
        batch_reductions = OffsetVector{T}(undef, 1:length(batches))
    else
        batch_reductions = OffsetVector{T}(undef, 0:length(batches))
        batch_reductions[0] = init
    end
    MapReduceWorkspace(red, f, batches, arena_src_views, batch_reductions)
end

@inline function get_return_type(red, f, srcs)
    Core.Compiler.return_type(red, Tuple{typeof(f), typeof.(srcs)...})
end

@noinline function run!(o::MapReduceWorkspace)
    Threads.@threads for i in 1:length(o.batches)
        tid = Threads.threadid()
        src_views = o.arena_src_views[tid]
        inds = o.batches[i]
        for src_view in src_views
            src_view.eachindex = inds
        end
        o.batch_reductions[i] = o.reduction(o.f, src_views...)
    end
    o.reduction(identity, o.batch_reductions)
end
