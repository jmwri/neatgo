# neatgo

A Go implementation of [NeuroEvolution of Augmenting Topologies (NEAT)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

NEAT evolves both the weights *and* the structure of a neural network. It starts
from a minimal topology and grows it, using historical markings to line genes up
during crossover and speciation to protect new structure while it is still bad
at its job.

## Install

```
go get github.com/jmwri/neatgo/v2
```

```go
import (
    "github.com/jmwri/neatgo/v2/neat"
    "github.com/jmwri/neatgo/v2/network"
)
```

v2 is a substantial rewrite and is not source compatible with v1. See
[UPGRADING.md](UPGRADING.md) if you are coming from v1.0.0.

## Quick start

Write a function that scores one network. The library runs it across the whole
population in parallel.

```go
func evaluate(_ context.Context, net *network.Network) (float64, error) {
    output := make([]float64, net.NumOutputs())

    fitness := 0.0
    for i, input := range xorInputs {
        if err := net.ActivateInto(input, output); err != nil {
            return 0, err
        }
        fitness += 1 - math.Pow(output[0]-xorAnswers[i], 2)
    }
    return fitness, nil
}

func main() {
    cfg := neat.DefaultConfig(2, 1) // 2 inputs, 1 output; hidden structure is evolved
    cfg.PopulationSize = 150
    cfg.Seed = 12345 // omit for a random seed, recorded on pop.Seed

    pop, err := neat.GeneratePopulation(cfg)
    if err != nil {
        log.Fatal(err)
    }

    pop, err = neat.Run(context.Background(), pop, evaluate, neat.RunOptions{
        MaxGenerations: 300,
        Solved: func(pop neat.Population) bool {
            return pop.BestGenomeFitness >= 3.9
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    best, err := pop.BestEverGenome.Compile()
    if err != nil {
        log.Fatal(err)
    }
    output, err := best.Activate([]float64{1, 0})
}
```

See `example/xor` for a complete program. It solves XOR in around 32
generations on average, and in 300 out of 300 seeded runs within 80.

## Concurrency

Every genome in a generation is compiled and evaluated concurrently. Work is
pulled from a shared counter rather than split up front, so a population of
wildly different genome sizes still spreads evenly across the workers.

`Config.Parallelism` controls how many run at once:

| Value | Behaviour |
| --- | --- |
| `0` (default) | One worker per CPU. The right choice when the evaluator is CPU bound. |
| `n` | Exactly `n` workers. |
| `neat.Unlimited` | One goroutine per genome. Use it when the evaluator spends its time blocked - on a simulator, a subprocess, a network call - rather than computing. |

Breeding the next generation is parallel too, but the structural mutations that
follow it are applied one genome at a time: they draw historical markings from
a shared registry, and racing for those would make the gene numbering - and so
the whole run - depend on which goroutine won.

Evaluation scales close to linearly. Measured on 16 logical cores, 150 genomes,
200 activations each:

```
workers=1            8.06 ms/generation
workers=2            4.50 ms   1.8x
workers=4            2.56 ms   3.1x
workers=8            2.00 ms   4.0x
workers=16           1.58 ms   5.1x
workers=unlimited    1.45 ms   5.6x
```

Things worth knowing:

- **The evaluator is called concurrently.** It must not write to shared state
  without synchronisation. Anything it needs per-call - an output buffer, a
  simulator - should be created inside it.
- **A compiled `*network.Network` is immutable and safe to share.** `Activate`
  may be called from any number of goroutines at once. `ActivateInto` is too,
  provided each goroutine owns its output slice. A recurrent network is run
  with `Step` and a `Memory` the evaluator creates with `net.NewMemory()`; the
  memory is what makes one network safe to run on many goroutines at once.
- **Errors and cancellation stop the generation.** An evaluator returning an
  error, or a cancelled context, cancels the remaining workers and surfaces the
  cause. The evaluator receives the context, so a long evaluation can bail out
  partway rather than only between genomes.
- **The population is always returned**, including when a run ends early, so the
  best genome found so far survives an error or a cancellation.

## Reproducibility

Every random choice comes from one seeded source, so a run is repeatable:

```go
cfg.Seed = 12345
```

Leave `Seed` at zero and one is drawn for you and recorded on `pop.Seed`, so a
run worth keeping can be replayed by feeding that value back in.

The result does not depend on how the work is scheduled. Each generation gets a
random stream derived from the seed and the generation number, and each
offspring a stream derived from that, so `Parallelism` changes how fast a run
goes and nothing else. There is a test asserting the same seed gives the same
population at 1, 2, 4, 8, 16 and unlimited workers.

`Config` is a plain description of what to do and is safe to copy. The mutable
run state - the random source and the innovation registry - lives on
`pop.Breeder`, which is why building two populations from one `Config` gives two
independent runs rather than one silently continuing the other's gene numbering.

## Saving a trained network

`Genome` is a tree of exported types, so it round-trips through `encoding/json`
with no custom marshalling:

```go
data, err := json.Marshal(pop.BestEverGenome)
// ... later, in another process ...
var genome neat.Genome
err = json.Unmarshal(data, &genome)
net, err := genome.Compile()
```

## Checkpointing a run

`Population.Snapshot()` captures everything a run needs to carry on - genomes,
species, and the innovation registry - as plain exported types:

```go
data, err := json.Marshal(pop.Snapshot())
// ... after a crash ...
var snapshot neat.Snapshot
err = json.Unmarshal(data, &snapshot)
pop, err := neat.Restore(snapshot)
```

A resumed run continues exactly as the uninterrupted one would have. There is no
random state in the snapshot because a generation's randomness is derived from
the seed and the generation number, so picking up at generation 12 replays the
same stream generation 12 would have had.

## Compiling a network

`Compile` resolves the evaluation order, each node's activation function and
every connection's source once. `Activate` is then a flat loop over slices.

```go
net, err := genome.Compile()
output, err := net.Activate(input)          // allocates the output slice
err = net.ActivateInto(input, output)       // reuses yours; zero allocations
```

Compiling costs about the same as a few dozen activations, so compile once and
reuse. `network.Activate(nodes, connections, input)` is available for one-off
use but compiles on every call.

`net.NumNodes()` and `net.NumConnections()` report the size of the *running*
network, with disabled connections already dropped - which is what a complexity
penalty should be built on.

## Custom evaluation

`RunGeneration` is `Evaluate` followed by `Advance`. Call them separately when
scoring cannot be expressed as one independent evaluator per genome, such as a
competitive tournament:

```go
nets := make([]*network.Network, len(pop.Genomes))
for i, genome := range pop.Genomes {
    nets[i], err = genome.Compile()
}

// ... play genomes off against each other, writing into pop.GenomeFitness ...

pop = neat.Advance(pop)
```

## How a generation runs

```
evaluate      every genome is compiled and scored concurrently
record best   the fittest raw result of this generation is recorded
speciate      genomes are grouped by compatibility distance to a representative
retarget      the compatibility threshold is nudged towards TargetSpecies
rank          genomes within species, and species by their best current member
share         adjusted fitness = raw fitness / species size
prune         species that have not improved for too long are removed
cull          only the top SurvivalThreshold of each species may reproduce
reproduce     offspring are allocated per species in proportion to the sum of
              its adjusted fitness, which is its mean raw fitness; parents are
              picked among the survivors by roulette over fitness
```

## Design notes

**Historical markings.** Genes are identified by an innovation number handed out
by `neat.Innovations`, not by a bare counter. The same structural change - a
connection between two given nodes, or the split of a given connection - always
receives the same ID no matter which genome discovers it. This is what makes
crossover able to align two genomes and what makes the compatibility distance
mean something.

**One starting genotype.** `GeneratePopulation` builds a single template genome
and copies it, drawing fresh weights for each member. Generating each genome
independently would give them disjoint gene IDs, and the population would
fragment into one species per genome before evolution had run a step.

**Neutral add-node mutations.** Splitting a connection gives the incoming half a
weight of 1 and the outgoing half the original weight, so a new node starts out
reproducing roughly what it replaced. New structure then has time to be
optimised rather than being killed off on arrival.

**Layers.** Genomes carry explicit layers. Feed-forward, connections only ever
run from an earlier layer to a later one, which guarantees the network is
acyclic and can be evaluated in a single pass; skip connections across any
number of layers are allowed. With `Config.Recurrent` set, mutation may also
wire a connection backwards, within a layer, or from a node to itself. Such a
connection reads the value its source held at the end of the previous
activation, so the network has memory: its answer can depend on what it has
already seen. That is worth having for anything sequential and worth avoiding
otherwise, since it enlarges the search space and makes an evaluation depend
on the order it happened in. A recurrent genome compiles with
`CompileRecurrent` and runs with `Step`; `Config.Recurrent` makes `Run` do
both.

**Node genes carry parameters.** Unlike the paper, where only connections have
weights, each hidden and output node has its own bias and activation function,
and both take part in the compatibility distance. Input and bias nodes are
structural: an input passes its value through untouched, a bias node emits a
constant 1, and neither is ever mutated. Because every hidden and output node
already has a bias, a separate bias node adds nothing a genome cannot express,
only more connections to fit, so `BiasNodes` defaults to zero.

## Configuration

`neat.DefaultConfig(layerSizes...)` returns sensible defaults. The settings that
most affect a run:

| Setting | Default | Notes |
| --- | --- | --- |
| `PopulationSize` | 150 | Genomes per generation. |
| `Parallelism` | 0 | Concurrent evaluations. 0 is one per CPU. |
| `Recurrent` | false | Allow connections that loop back, giving the network memory. |
| `BiasNodes` | 0 | Explicit bias inputs. Hidden and output nodes carry a bias of their own. |
| `Seed` | 0 | Fixes the run's random sequence. 0 draws one and records it on `pop.Seed`. |
| `AddNodeMutationRate` | 0.03 | Structural mutations should be rare, so topology grows slowly. |
| `AddConnectionMutationRate` | 0.08 | Keep above the matching delete rate or structure erodes. |
| `WeightMutationPower` | 0.5 | Std dev of the gaussian added when perturbing a weight. |
| `WeightInitStdDev` | 1 | Std dev a fresh weight is drawn from. `MinWeight`/`MaxWeight` only clamp. |
| `SpeciesCompatThreshold` | 3 | Lower means genomes must be more similar to share a species. |
| `TargetSpecies` | 10 | The threshold is retuned each generation to hit this. Set to 0 to disable. |
| `SpeciesStalenessThreshold` | 20 | Generations without improvement before a species is dropped. |
| `SurvivalThreshold` | 0.2 | Fraction of each species allowed to reproduce. |
| `Elitism` | 2 | Top genomes carried over per species unmutated. |

`GeneratePopulation` validates the config and reports every problem at once, so
a bad setting fails at the call rather than surfacing later as odd behaviour.
`Config.Validate()` can be called directly.

Errors are distinguishable with `errors.Is`: `neat.ErrInvalidConfig`,
`neat.ErrEvaluate`, `neat.ErrCompile`, `network.ErrCycle`,
`network.ErrUnknownActivation` and friends. An evaluator's own error is wrapped
rather than replaced, so `errors.Is` still finds it.

`RunGeneration` returns a `Population` whose `Cfg` may differ from the one you
passed in, because dynamic species targeting adjusts `SpeciesCompatThreshold`.
Always carry the returned `Population` forward.

## Known limitations

- The activation function registry is process-wide. Two populations in one
  process cannot use different implementations of the same activation name.
- `Species.Genomes` holds indices into `Population.Genomes`; reordering that
  slice by hand would silently corrupt species membership.

## Tests

```
go test ./...
go test ./neat/ -run XXX -bench .
go test ./network/ -run XXX -bench .
```
