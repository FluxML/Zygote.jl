module Forward

import ..Zygote
import ..Zygote: __new__

export pushforward

include("compiler.jl")
include("lib.jl")

pushforward(f, x...) = (ẋ...) -> _tangent((zerolike(f), ẋ...), f, x...)[2]

end
