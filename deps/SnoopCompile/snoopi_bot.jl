using SnoopCompile

snoopi_bot(
  BotConfig("Zygote"; os = ["linux", "windows", "macos"], version = [v"1.4.1", v"1.3.1"]),
  "$(@__DIR__)/example_script.jl",
)
