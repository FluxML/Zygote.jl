# Mostly copied over from Cassette in `src/overdub.jl`
# Return `Reflection` for signature `sigtypes` and `world`, if possible. Otherwise, return `nothing`.
function reflect(@nospecialize(sigtypes::Tuple), world::UInt = typemax(UInt))
    if length(sigtypes) > 2 && sigtypes[1] === typeof(invoke)
        @assert sigtypes[3] <: Type{<:Tuple}
        sigtypes = (sigtypes[2], sigtypes[3].parameters[1].parameters...)
    end
    # This works around a subtyping bug. Basically, callers can deconstruct upstream
    # `UnionAll` types in such a way that results in a type with free type variables, in
    # which case subtyping can just break.
    #
    # God help you if you try to use a type parameter here (e.g. `::Type{S} where S<:Tuple`)
    # instead of this nutty workaround, because the compiler can just rewrite `S` into
    # whatever it thinks is "type equal" to the actual provided value. In other words, if
    # `S` is defined as e.g. `f(::Type{S}) where S`, and you call `f(T)`, you should NOT
    # assume that `S === T`. If you did, SHAME ON YOU. It doesn't matter that such an
    # assumption holds true for essentially all other kinds of values. I haven't counted in
    # a while, but I'm pretty sure I have ~40+ hellish years of Julia experience, and this
    # still catches me every time. Who even uses this crazy language?
    S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
    (S.parameters[1]::DataType).name.module === Core.Compiler && return nothing
    _methods = Base._methods_by_ftype(S, -1, world)
    method_index = 0
    for i in 1:length(_methods)
        if _methods[i][1] === S
            method_index = i
            break
        end
    end
    method_index === 0 && return nothing
    type_signature, raw_static_params, method = _methods[method_index]
    if VERSION < v"1.8-"
        method_instance = Core.Compiler.specialize_method(method, type_signature, raw_static_params, false)
    else
        method_instance = Core.Compiler.specialize_method(method, type_signature, raw_static_params; preexisting=false)
    end
    method_signature = method.sig
    static_params = Any[raw_static_params...]
    return method_instance, method_signature, static_params
end


# ugly hack to make differentiating `getproperty` infer a lot better
@generated function _pullback(cx::AContext, ::typeof(literal_getproperty), x, ::Val{f}) where f
    sig(x) = Tuple{x, typeof(f)}
    rrule_sig(x) = Tuple{typeof(getproperty), x, typeof(f)}
    pb_sig(x) = Tuple{cx, typeof(getproperty), x, typeof(f)}

    # either `getproperty` has a custom implementation or `_pullback(cx, getproperty, x, f)`
    # / `rrule(getproperty, x, f) is overloaded directly
    is_getfield_fallback = which(getproperty, sig(x)) == which(getproperty, sig(Any)) &&
        which(_pullback, pb_sig(x)) == which(_pullback, pb_sig(Any)) &&
        which(rrule, rrule_sig(x)) == which(rrule, rrule_sig(Any))

    #ccall(:jl_safe_printf, Cvoid, (Cstring,), "$is_getfield_fallback: $x\n")

    if is_getfield_fallback
        # just copy pullback of `literal_getfield`
        mi, _sig, sparams = reflect((typeof(_pullback), cx, typeof(literal_getfield), x, Val{f}))
        ci = copy(Core.Compiler.retrieve_code_info(mi))

        # we need to change the second arg to `_pullback` from `literal_getproperty` to
        # `literal_getfield`
        Meta.partially_inline!(
            ci.code, Any[_pullback, Core.SlotNumber(2), literal_getfield],
            _sig, sparams, 0, 0, :propagate,
        )
        ci.inlineable = true

        # backedge for `_pullback`, see https://docs.julialang.org/en/v1/devdocs/ast/#MethodInstance
        # this will cause a backedge to this particular MethodInstance to be attached to
        # `_pullback(cx, getproperty, x, f)`
        mi_pb_getproperty, _, _ = reflect((typeof(_pullback), pb_sig(x).parameters...))
        mi_getproperty, _, _ = reflect((typeof(getproperty), sig(x).parameters...))
        mi_rrule, _, _ = reflect((typeof(rrule), rrule_sig(x).parameters...))
        ci.edges = Core.MethodInstance[mi, mi_pb_getproperty, mi_getproperty, mi_rrule]

        return ci
    else
        # nothing to optimize here, need to recurse into `getproperty`
        return quote
            Base.@_inline_meta
            _pullback(cx, getproperty, x, $(QuoteNode(f)))
        end
    end
end
