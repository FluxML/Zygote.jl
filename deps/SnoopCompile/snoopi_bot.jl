using SnoopCompile

@snoopi_bot BotConfig("Zygote") begin
  # we should hide them in a file, so Julia doesn't exapnd the macros before executing `using Zygote`
  using Zygote
  zygote_rootpath = dirname(dirname(pathof(Zygote)))
  include("$zygote_rootpath/deps/SnoopCompile/example_script.jl")
end
