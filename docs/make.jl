using Documenter, Zygote

makedocs(sitename="My Documentation")

deploydocs(
    repo = "github.com/FluxML/Zygote.jl.git",
)
