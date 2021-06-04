module Forward

import ..Zygote
import ..Zygote: __new__, __splatnew__, Numeric

export pushforward

include("compiler.jl")
include("interface.jl")
include("lib.jl")
include("number.jl")
include("array.jl")
include("broadcast.jl")

end
