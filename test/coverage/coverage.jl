# only push coverage from one bot
get(ENV, "TRAVIS_OS_NAME", nothing)       == "linux" || exit(0)
get(ENV, "TRAVIS_JULIA_VERSION", nothing) == "1.1"   || exit(0)

using Coverage

cd(joinpath(@__DIR__, "..", "..")) do
    Codecov.submit(Codecov.process_folder())
end
