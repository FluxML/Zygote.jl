module Compiler

@nospecialize
@static if VERSION > v"1.5.0-"
  Base.Experimental.@optlevel 0
end

import ..Zygote
using IRTools, MacroTools

include("reverse.jl")
include("emit.jl")

end

import .Compiler: _lookup_grad
