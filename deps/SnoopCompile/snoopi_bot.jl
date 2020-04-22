using SnoopCompile

@snoopi_bot BotConfig("Zygote") begin
  # we should hide them in a file, so Julia doesn't exapnd the macros before executing `using Zygote`
  include("example_script.jl")
end
