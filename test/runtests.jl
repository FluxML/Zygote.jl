using Zygote
using ReTestItems
using CUDA

testitem_timeout = 30 * 60 # 30 minutes
nworkers = 4
nworker_threads = 1

run_cuda = !haskey(ENV, "GITHUB_ACTION") && CUDA.has_cuda()
run_cuda || @warn "CUDA not found - Skipping CUDA Tests"

runtests(Zygote; nworkers, nworker_threads, testitem_timeout) do ti
    if run_cuda
        return true
    else
        return ti.name != "cuda"
    end
end
