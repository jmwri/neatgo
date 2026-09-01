package neat_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
)

// BenchmarkGeneration measures a whole generation: evaluation plus speciation,
// selection and reproduction. The evaluator is deliberately cheap so that the
// library's own overhead is what shows up.
func BenchmarkGeneration(b *testing.B) {
	run := func(b *testing.B, popSize int, warmup int) {
		cfg := neat.DefaultConfig(8, 4)
		cfg.PopulationSize = popSize
		pop, err := neat.GeneratePopulation(cfg)
		if err != nil {
			b.Fatal(err)
		}

		eval := func(_ context.Context, net *network.Network) (float64, error) {
			input := make([]float64, net.NumInputs())
			output := make([]float64, net.NumOutputs())
			fitness := .0
			for i := 0; i < 8; i++ {
				for j := range input {
					input[j] = math.Sin(float64(i + j))
				}
				if err := net.ActivateInto(input, output); err != nil {
					return 0, err
				}
				fitness += output[0]
			}
			return fitness, nil
		}
		ctx := context.Background()

		// Let the population grow some structure first, so we are not
		// benchmarking the trivial two-layer starting genome.
		for i := 0; i < warmup; i++ {
			pop, err = neat.RunGeneration(ctx, pop, eval)
			if err != nil {
				b.Fatal(err)
			}
		}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			pop, err = neat.RunGeneration(ctx, pop, eval)
			if err != nil {
				b.Fatal(err)
			}
		}
	}

	b.Run("pop=150/warm=0", func(b *testing.B) { run(b, 150, 0) })
	b.Run("pop=150/warm=100", func(b *testing.B) { run(b, 150, 100) })
	b.Run("pop=1000/warm=50", func(b *testing.B) { run(b, 1000, 50) })
}

// BenchmarkAdvance isolates the non-evaluation half of a generation:
// speciation, fitness sharing, selection and reproduction. Parallelism here
// affects only breeding, since no evaluator runs.
func BenchmarkAdvance(b *testing.B) {
	for _, parallelism := range []int{1, 2, 4, 8, 0} {
		name := fmt.Sprintf("workers=%d", parallelism)
		if parallelism == 0 {
			name = "workers=default"
		}
		b.Run(name, func(b *testing.B) {
			cfg := neat.DefaultConfig(8, 4)
			cfg.PopulationSize = 500
			cfg.Seed = 99
			cfg.Parallelism = parallelism
			pop, err := neat.GeneratePopulation(cfg)
			if err != nil {
				b.Fatal(err)
			}
			score := func(i int) {
				for j := range pop.GenomeFitness {
					pop.GenomeFitness[j] = float64((i + j) % 17)
				}
			}
			score(0)
			// Grow structure so speciation has real work to do.
			for i := 0; i < 60; i++ {
				pop = neat.Advance(pop)
				score(i)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				pop = neat.Advance(pop)
				score(i)
			}
		})
	}
}
