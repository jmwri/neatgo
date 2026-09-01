package neat_test

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fingerprint captures enough of a population to tell two runs apart, down to
// the exact bit pattern of every weight and bias.
func fingerprint(pop neat.Population) string {
	var b strings.Builder
	for _, genome := range pop.Genomes {
		for _, node := range genome.Layers.Nodes() {
			fmt.Fprintf(&b, "n%d:%x:%s;", node.ID, math.Float64bits(node.Bias), node.ActivationFn)
		}
		for _, connection := range genome.Connections {
			fmt.Fprintf(&b, "c%d:%d>%d:%x:%t;",
				connection.ID, connection.From, connection.To,
				math.Float64bits(connection.Weight), connection.Enabled)
		}
		b.WriteString("|")
	}
	return b.String()
}

// A run with a fixed seed must be repeatable end to end. Without this, an
// interesting result cannot be reproduced and a rare failure cannot be bisected.
func TestRun_IsReproducibleForAFixedSeed(t *testing.T) {
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}

	runOnce := func() neat.Population {
		cfg := neat.DefaultConfig(2, 1)
		cfg.PopulationSize = 60
		cfg.Seed = 20260901
		// Deliberately parallel: evaluation order must not affect the outcome.
		cfg.Parallelism = neat.Unlimited

		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)

		pop, err = neat.Run(context.Background(), pop, func(_ context.Context, net *network.Network) (float64, error) {
			fitness := .0
			for _, input := range inputs {
				output, err := net.Activate(input)
				if err != nil {
					return 0, err
				}
				fitness += output[0]
			}
			return fitness, nil
		}, neat.RunOptions{MaxGenerations: 25})
		require.NoError(t, err)
		return pop
	}

	first := runOnce()
	second := runOnce()

	assert.Equal(t, first.Seed, second.Seed)
	assert.Equal(t, first.Generation, second.Generation)
	assert.Equal(t, first.BestEverGenomeFitness, second.BestEverGenomeFitness)
	assert.Equal(t, len(first.Species), len(second.Species))
	assert.Equal(t, fingerprint(first), fingerprint(second), "identical seeds must produce identical populations")
}

// Different seeds must actually diverge, otherwise the seed is not doing
// anything.
func TestRun_DifferentSeedsDiverge(t *testing.T) {
	runOnce := func(seed uint64) neat.Population {
		cfg := neat.DefaultConfig(2, 1)
		cfg.PopulationSize = 40
		cfg.Seed = seed
		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)
		pop, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{MaxGenerations: 10})
		require.NoError(t, err)
		return pop
	}

	assert.NotEqual(t, fingerprint(runOnce(1)), fingerprint(runOnce(2)))
}

// A zero seed means "pick one for me", but the choice is recorded so the run
// can be replayed.
func TestGeneratePopulation_RecordsAGeneratedSeed(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 10
	cfg.Seed = 0

	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	assert.NotZero(t, pop.Seed, "a generated seed must be recorded so the run can be replayed")

	// Replaying with the recorded seed reproduces the starting population.
	cfg.Seed = pop.Seed
	replay, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	assert.Equal(t, fingerprint(pop), fingerprint(replay))
}

// Breeding runs in parallel, so the result must not depend on how many workers
// happened to do it. If it did, a seeded run would still not be reproducible
// across machines with different core counts.
func TestRun_IsIndependentOfWorkerCount(t *testing.T) {
	runWith := func(parallelism int) neat.Population {
		cfg := neat.DefaultConfig(3, 2)
		cfg.PopulationSize = 120
		cfg.Seed = 4242
		cfg.Parallelism = parallelism

		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)

		var n int
		pop, err = neat.Run(context.Background(), pop, func(_ context.Context, net *network.Network) (float64, error) {
			output, err := net.Activate([]float64{1, -1, .5})
			if err != nil {
				return 0, err
			}
			n++
			return output[0] + output[1], nil
		}, neat.RunOptions{MaxGenerations: 30})
		require.NoError(t, err)
		return pop
	}

	want := fingerprint(runWith(1))
	for _, parallelism := range []int{2, 4, 8, 16, neat.Unlimited} {
		assert.Equal(t, want, fingerprint(runWith(parallelism)),
			"parallelism %d produced a different run", parallelism)
	}
}
