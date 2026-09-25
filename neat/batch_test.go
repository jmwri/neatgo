package neat_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var xorData = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
var xorWant = []float64{0, 1, 1, 0}

func xorScore(_ *network.Network, out network.Outputs) (float64, error) {
	fitness := 0.0
	for i := 0; i < out.Len(); i++ {
		fitness += 1 - math.Pow(out.Sample(i)[0]-xorWant[i], 2)
	}
	return fitness, nil
}

func TestEvaluateBatch_MatchesEvaluate(t *testing.T) {
	// Grow some structure first so the population is not one shape.
	pop := testPopulation(t, 80)
	for i := 0; i < 5; i++ {
		var err error
		pop, err = neat.RunGeneration(context.Background(), pop, xorEvaluator())
		require.NoError(t, err)
	}

	want := pop
	want.GenomeFitness = make([]float64, len(pop.Genomes))
	require.NoError(t, neat.Evaluate(context.Background(), want, xorEvaluator()))

	got := pop
	got.GenomeFitness = make([]float64, len(pop.Genomes))
	require.NoError(t, neat.EvaluateBatch(context.Background(), got, neat.Batch{
		Activator: neat.CPUActivator{},
		Inputs:    xorData,
		Score:     xorScore,
	}))

	assert.Equal(t, want.GenomeFitness, got.GenomeFitness)
}

func TestRunBatch_SolvesXOR(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.Seed = 1
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	pop, err = neat.RunBatch(context.Background(), pop, neat.Batch{
		Activator: neat.CPUActivator{},
		Inputs:    xorData,
		Score:     xorScore,
	}, neat.RunOptions{
		MaxGenerations: 300,
		Solved:         func(p neat.Population) bool { return p.BestGenomeFitness >= 3.9 },
	})
	require.NoError(t, err)
	assert.GreaterOrEqual(t, pop.BestEverGenomeFitness, 3.9)
}

func TestRunBatch_MatchesRunForTheSameSeed(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.Seed = 5
	opts := neat.RunOptions{MaxGenerations: 15}

	a, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	a, err = neat.Run(context.Background(), a, xorEvaluator(), opts)
	require.NoError(t, err)

	b, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	b, err = neat.RunBatch(context.Background(), b, neat.Batch{Activator: neat.CPUActivator{}, Inputs: xorData, Score: xorScore}, opts)
	require.NoError(t, err)

	assert.Equal(t, a.BestEverGenomeFitness, b.BestEverGenomeFitness)
	assert.Equal(t, a.Genomes, b.Genomes)
}

func TestEvaluateBatch_Errors(t *testing.T) {
	ctx := context.Background()
	good := neat.Batch{Activator: neat.CPUActivator{}, Inputs: xorData, Score: xorScore}

	t.Run("missing pieces", func(t *testing.T) {
		pop := testPopulation(t, 5)
		assert.ErrorIs(t, neat.EvaluateBatch(ctx, pop, neat.Batch{Inputs: xorData, Score: xorScore}), neat.ErrNoEvaluator)
		assert.ErrorIs(t, neat.EvaluateBatch(ctx, pop, neat.Batch{Activator: good.Activator, Inputs: xorData}), neat.ErrNoEvaluator)
		assert.ErrorIs(t, neat.EvaluateBatch(ctx, pop, neat.Batch{Activator: good.Activator, Score: xorScore}), neat.ErrEmptyBatch)
	})

	t.Run("recurrent population", func(t *testing.T) {
		cfg := neat.DefaultConfig(2, 1)
		cfg.Recurrent = true
		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)
		assert.ErrorIs(t, neat.EvaluateBatch(ctx, pop, good), neat.ErrRecurrentBatch)
	})

	t.Run("wrong input size", func(t *testing.T) {
		pop := testPopulation(t, 5)
		b := good
		b.Inputs = [][]float64{{1, 2, 3}}
		err := neat.EvaluateBatch(ctx, pop, b)
		assert.ErrorIs(t, err, neat.ErrBatch)
		assert.ErrorIs(t, err, network.ErrInputSize)
	})

	t.Run("scorer error is wrapped", func(t *testing.T) {
		pop := testPopulation(t, 5)
		boom := errors.New("boom")
		b := good
		b.Score = func(*network.Network, network.Outputs) (float64, error) { return 0, boom }
		err := neat.EvaluateBatch(ctx, pop, b)
		assert.ErrorIs(t, err, neat.ErrEvaluate)
		assert.ErrorIs(t, err, boom)
	})

	t.Run("cancelled context", func(t *testing.T) {
		pop := testPopulation(t, 5)
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		assert.ErrorIs(t, neat.EvaluateBatch(cancelled, pop, good), context.Canceled)
	})

	t.Run("activator returning the wrong shape", func(t *testing.T) {
		pop := testPopulation(t, 5)
		b := good
		b.Activator = shortActivator{}
		assert.ErrorIs(t, neat.EvaluateBatch(ctx, pop, b), neat.ErrBatch)
	})

	t.Run("failed generation does not advance", func(t *testing.T) {
		pop := testPopulation(t, 5)
		b := good
		b.Inputs = [][]float64{{1}}
		next, err := neat.RunGenerationBatch(ctx, pop, b)
		require.Error(t, err)
		assert.Equal(t, pop.Generation, next.Generation)
	})
}

type shortActivator struct{}

func (shortActivator) ActivateBatch(nets []*network.Network, inputs [][]float64) (network.BatchOutputs, error) {
	return network.NewBatchOutputs(len(nets)-1, len(inputs), 1), nil
}
