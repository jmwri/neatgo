package neat_test

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// constantFitness scores every genome the same.
func constantFitness(f float64) neat.Evaluator {
	return func(context.Context, *network.Network) (float64, error) {
		return f, nil
	}
}

// The population must stay exactly the size it was configured with. Letting it
// drift means the offspring allocation, elitism and species minimums are
// fighting each other.
func TestRunGeneration_PopulationSizeIsStable(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 60
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	for generation := 0; generation < 30; generation++ {
		pop, err = neat.RunGeneration(context.Background(), pop, constantFitness(1))
		require.NoError(t, err)

		assert.Len(t, pop.Genomes, 60, "generation %d", generation)
		assert.Len(t, pop.GenomeFitness, 60, "generation %d", generation)
		assert.Len(t, pop.GenomeAdjustedFitness, 60, "generation %d", generation)

		for i, species := range pop.Species {
			assert.NotEmpty(t, species.Genomes, "species %d has no members", i)
			for _, genomeIndex := range species.Genomes {
				assert.Less(t, genomeIndex, len(pop.Genomes), "species %d references a genome that does not exist", i)
			}
		}
	}
}

// Reported fitness must be the raw fitness of a genome that was actually
// evaluated this generation, not a fitness-shared value or an unevaluated
// member of the generation that is about to run.
func TestRunGeneration_ReportsRawBestFitness(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 30
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	// Exactly one genome scores 10, everyone else scores 1.
	var seen atomic.Int64
	pop, err = neat.RunGeneration(context.Background(), pop, func(context.Context, *network.Network) (float64, error) {
		if seen.Add(1) == 1 {
			return 10, nil
		}
		return 1, nil
	})
	require.NoError(t, err)

	assert.Equal(t, 10.0, pop.BestGenomeFitness)
	assert.Equal(t, 10.0, pop.BestEverGenomeFitness)
}

// A single non-finite score must not be able to poison selection for every
// other genome.
func TestRunGeneration_SanitisesNonFiniteFitness(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 20
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	var seen atomic.Int64
	pop, err = neat.RunGeneration(context.Background(), pop, func(context.Context, *network.Network) (float64, error) {
		switch seen.Add(1) {
		case 1:
			return math.Inf(-1), nil
		case 2:
			return math.NaN(), nil
		}
		return 5, nil
	})
	require.NoError(t, err)

	for i, fitness := range pop.GenomeFitness {
		assert.False(t, math.IsInf(fitness, 0), "genome %d has a non-finite fitness", i)
		assert.False(t, math.IsNaN(fitness), "genome %d has a NaN fitness", i)
	}
	assert.Len(t, pop.Genomes, 20)
}

func TestRunGeneration_ReturnsEvaluatorError(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 20
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	boom := errors.New("boom")
	before := pop.Generation
	pop, err = neat.RunGeneration(context.Background(), pop, func(context.Context, *network.Network) (float64, error) {
		return 0, boom
	})

	assert.ErrorIs(t, err, boom)
	assert.Equal(t, before, pop.Generation, "a failed generation must not advance the population")
}

func TestRun_StopsWhenSolved(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 20
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	generations := 0
	pop, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{
		MaxGenerations: 100,
		Solved:         func(pop neat.Population) bool { return pop.Generation >= 5 },
		OnGeneration: func(neat.Population) error {
			generations++
			return nil
		},
	})

	require.NoError(t, err)
	assert.Equal(t, 5, generations)
	assert.Equal(t, 5, pop.Generation)
}

func TestRun_StopsAtMaxGenerations(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 20
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	pop, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{MaxGenerations: 7})
	require.NoError(t, err)
	assert.Equal(t, 7, pop.Generation)
}

func TestRun_StopsOnCallbackError(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 20
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	stop := errors.New("checkpoint failed")
	pop, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{
		MaxGenerations: 100,
		OnGeneration: func(pop neat.Population) error {
			if pop.Generation == 3 {
				return stop
			}
			return nil
		},
	})

	assert.ErrorIs(t, err, stop)
	// The population is still returned, so work done up to the failure is kept.
	assert.Equal(t, 3, pop.Generation)
	assert.Len(t, pop.Genomes, 20)
}

// A Run with no stopping condition at all would spin forever. Say so rather
// than hanging.
func TestRun_RejectsUnterminatedRun(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 10
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	_, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{})
	assert.ErrorContains(t, err, "never terminate")
}

func TestRun_StopsOnContextCancellation(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 20
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	pop, err = neat.Run(ctx, pop, constantFitness(1), neat.RunOptions{
		MaxGenerations: 1000,
		OnGeneration: func(pop neat.Population) error {
			if pop.Generation == 4 {
				cancel()
			}
			return nil
		},
	})
	defer cancel()

	assert.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, 4, pop.Generation)
}

// End to end: the whole pipeline must actually be able to learn something that
// a network without a hidden node cannot represent.
func TestEvolution_LearnsXOR(t *testing.T) {
	if testing.Short() {
		t.Skip("evolutionary run")
	}

	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 150
	cfg.BiasNodes = 0
	cfg.OutputActivationFn = network.Sigmoid
	cfg.HiddenActivationFns = []network.ActivationFunctionName{network.Sigmoid}

	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	answers := []float64{0, 1, 1, 0}

	pop, err = neat.Run(context.Background(), pop, func(_ context.Context, net *network.Network) (float64, error) {
		fitness := .0
		for i, input := range inputs {
			output, err := net.Activate(input)
			if err != nil {
				return 0, err
			}
			fitness += 1 - math.Pow(output[0]-answers[i], 2)
		}
		return fitness, nil
	}, neat.RunOptions{
		MaxGenerations: 250,
		Solved:         func(pop neat.Population) bool { return pop.BestEverGenomeFitness >= 3.9 },
	})
	require.NoError(t, err)

	assert.GreaterOrEqual(t, pop.BestEverGenomeFitness, 3.9, "failed to learn xor")

	// And the winner really does classify all four cases.
	best, err := pop.BestEverGenome.Compile()
	require.NoError(t, err)
	for i, input := range inputs {
		output, err := best.Activate(input)
		require.NoError(t, err)
		assert.Equal(t, answers[i], math.Round(output[0]), "input %v", input)
	}
}

// Advance lets callers score genomes any way they like - here a round robin
// tournament, which cannot be expressed as one independent Evaluator per
// genome - and still use the rest of the pipeline.
func TestAdvance_AcceptsExternallyScoredFitness(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 24
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	for generation := 0; generation < 5; generation++ {
		nets := make([]*network.Network, len(pop.Genomes))
		for i, genome := range pop.Genomes {
			nets[i], err = genome.Compile()
			require.NoError(t, err)
		}

		// Every genome plays every other; a win is scored by producing the
		// larger output.
		for i := range nets {
			pop.GenomeFitness[i] = 0
		}
		for i := range nets {
			for j := i + 1; j < len(nets); j++ {
				a, err := nets[i].Activate([]float64{1, 0})
				require.NoError(t, err)
				b, err := nets[j].Activate([]float64{1, 0})
				require.NoError(t, err)
				if a[0] > b[0] {
					pop.GenomeFitness[i]++
				} else {
					pop.GenomeFitness[j]++
				}
			}
		}

		pop = neat.Advance(pop)
		assert.Equal(t, generation+1, pop.Generation)
		assert.Len(t, pop.Genomes, 24)
	}
	assert.Greater(t, pop.BestEverGenomeFitness, 0.0)
}
