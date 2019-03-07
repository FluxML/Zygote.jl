using Documenter, Zygote

makedocs(
  sitename="Zygote",
  pages = [
        "Home" => "index.md",
        "Custom Adjoints" => "adjoints.md",
        "Profiling" => "profiling.md",
        "Internals" => "internals.md"])

deploydocs(
    repo = "github.com/FluxML/Zygote.jl.git",
)
