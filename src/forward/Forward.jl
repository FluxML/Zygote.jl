module Forward

import ..Zygote
import ..Zygote: __new__

export pushforward

include("compiler.jl")
include("interface.jl")
include("lib.jl")

end
