module Forward

import ..Zygote
import ..Zygote: __new__, __splatnew__

export pushforward

include("compiler.jl")
include("interface.jl")
include("lib.jl")

end
