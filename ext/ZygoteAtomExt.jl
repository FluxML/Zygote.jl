module ZygoteAtomExt

using Atom
using Zygote
using Zygote.Profile

Zygote.Profile.atom_expandpath(path::String) = Atom.expandpath(path)
Zygote.Profile.juno(node::Zygote.Profile.Node) = Atom.msg("profile", Zygote.Profile.tojson(node))

end
