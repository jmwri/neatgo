# Upgrading from v1 to v2

v2 is a rewrite. The evolutionary algorithm in v1 did not work - a population
never mutated for its first fifteen generations and could not solve XOR - so
correctness fixes were not separable from the API. Everything below is a
breaking change.

Start with `go get github.com/jmwri/neatgo/v2` and change imports to
`github.com/jmwri/neatgo/v2/...`.

## Evaluating a population

v1 required you to launch a goroutine per genome *before* calling
`RunGeneration`, then drive four unbuffered channels per genome in exactly the
right order. Any deviation deadlocked.

```go
// v1
clientStates := pop.States()
wg := sync.WaitGroup{}
wg.Add(len(clientStates))
for _, state := range clientStates {
    go func(state neat.ClientGenomeState) {
        defer wg.Done()
        for _, input := range inputs {
            state.SendInput() <- input
            select {
            case output := <-state.GetOutput():
                fitness += score(output)
            case err := <-state.GetError():
                log.Println(err)
            }
        }
        close(state.SendInput())
        state.SendFitness() <- fitness
        close(state.SendFitness())
    }(state)
}
pop = neat.RunGeneration(pop)
wg.Wait()
```

In v2 you write a function that scores one network, and the library runs it
across the population in parallel.

```go
// v2
func evaluate(ctx context.Context, net *network.Network) (float64, error) {
    fitness := 0.0
    for _, input := range inputs {
        output, err := net.Activate(input)
        if err != nil {
            return 0, err
        }
        fitness += score(output)
    }
    return fitness, nil
}

pop, err := neat.Run(ctx, pop, evaluate, neat.RunOptions{
    MaxGenerations: 300,
    Solved:         func(pop neat.Population) bool { return pop.BestGenomeFitness >= target },
})
```

`Population.States`, `ClientGenomeState`, `BackendGenomeState` and
`GenomeState` are gone.

If your scoring cannot be expressed as one independent function per genome - a
competitive tournament, say - use `neat.Evaluate` and `neat.Advance` separately
instead of `neat.RunGeneration`.

## Running a network

Compile once and reuse. A `*network.Network` is immutable and safe to share
between goroutines.

```go
// v1
output, err := network.Activate(genome.Layers.Nodes(), genome.Connections, input)

// v2
net, err := genome.Compile()
output, err := net.Activate(input)
err = net.ActivateInto(input, output) // zero allocation
```

`network.Activate` still exists for one-off use but compiles on every call.

## Mutation and crossover

These are now methods on `*neat.Breeder`, which carries the settings, the random
source and the innovation registry together.

```go
// v1
genome, err := neat.GenerateGenome(cfg)
child := neat.MutateGenome(cfg, genome)
child = neat.Crossover(cfg, best, worst)

// v2
breeder := neat.NewBreeder(cfg, neat.NewRand(seed), nil)
genome, err := breeder.NewGenome()
child := breeder.MutateGenome(genome)
child = breeder.Crossover(best, worst)
```

A population's breeder is `pop.Breeder`. Build a Breeder *after* setting every
config field: it takes a copy of the config, so later edits to your `Config`
value are not seen.

## Config changes

Removed:

| v1 | replacement |
| --- | --- |
| `IDProvider`, `Innovations` | moved to `pop.Breeder`; a `Config` is now plain data and safe to copy |
| `RandFloatProvider`, `RandGaussianProvider` | `Seed`, plus an explicit `*neat.Rand` where needed |
| `TopGenomesFromSpeciesToFill` | offspring allocation handles this |
| `ResetOnExtinction` | was never implemented |

Added: `Seed`, `Parallelism`, `WeightInitStdDev`, `BiasInitStdDev`,
`EnabledMutationRate`, `MateDisabledRate`, `TargetSpecies`,
`SpeciesCompatThresholdAdjust`, `MinSpeciesCompatThreshold`.

Defaults changed substantially. `DefaultConfig` in v1 could not solve XOR;
in v2 it does, in around 32 generations. If you carried v1 values across, drop
them and start from `DefaultConfig`. Note that `BiasNodes` now defaults to 0:
hidden and output nodes carry their own bias, and an explicit bias node only
slows the search.

`GeneratePopulation` now validates the config and reports every problem at once.

## util package

`github.com/jmwri/neatgo/util` is now `internal` and no longer importable.

## Behaviour you should know about

- **Runs are reproducible.** Set `cfg.Seed`; the result does not depend on
  `Parallelism`. Leave it zero and the drawn seed is recorded on `pop.Seed`.
- **`pop.Cfg` may differ from the config you passed in**, because dynamic
  species targeting adjusts `SpeciesCompatThreshold`. Always carry the returned
  `Population` forward.
- **Errors are typed.** `errors.Is` against `neat.ErrInvalidConfig`,
  `neat.ErrEvaluate`, `network.ErrCycle` and friends.
- **Input and bias nodes are never mutated** and pass their signal through
  untouched. In v1 input nodes were given random biases and mutable activations,
  which corrupted the network's inputs.
