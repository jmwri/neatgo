package neat_test

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func xorEvaluator() neat.Evaluator {
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	answers := []float64{0, 1, 1, 0}
	return func(_ context.Context, net *network.Network) (float64, error) {
		fitness := .0
		for i, input := range inputs {
			output, err := net.Activate(input)
			if err != nil {
				return 0, err
			}
			fitness += 1 - math.Pow(output[0]-answers[i], 2)
		}
		return fitness, nil
	}
}

// A run interrupted and resumed from a snapshot must end up exactly where the
// uninterrupted run would have. Anything less and a checkpoint is a guess.
func TestSnapshot_ResumedRunMatchesUninterruptedRun(t *testing.T) {
	newPop := func() neat.Population {
		cfg := neat.DefaultConfig(2, 1)
		cfg.PopulationSize = 80
		cfg.Seed = 31337
		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)
		return pop
	}

	eval := xorEvaluator()
	ctx := context.Background()

	// Straight through, 30 generations.
	uninterrupted, err := neat.Run(ctx, newPop(), eval, neat.RunOptions{MaxGenerations: 30})
	require.NoError(t, err)

	// Stop at 12, serialise, restore, carry on to 30.
	partial, err := neat.Run(ctx, newPop(), eval, neat.RunOptions{MaxGenerations: 12})
	require.NoError(t, err)

	encoded, err := json.Marshal(partial.Snapshot())
	require.NoError(t, err)

	var snapshot neat.Snapshot
	require.NoError(t, json.Unmarshal(encoded, &snapshot))

	restored, err := neat.Restore(snapshot)
	require.NoError(t, err)
	assert.Equal(t, 12, restored.Generation)
	assert.Equal(t, partial.BestEverGenomeFitness, restored.BestEverGenomeFitness)

	resumed, err := neat.Run(ctx, restored, eval, neat.RunOptions{MaxGenerations: 30})
	require.NoError(t, err)

	assert.Equal(t, uninterrupted.Generation, resumed.Generation)
	assert.Equal(t, uninterrupted.BestEverGenomeFitness, resumed.BestEverGenomeFitness)
	assert.Equal(t, fingerprint(uninterrupted), fingerprint(resumed),
		"a resumed run must continue exactly where it left off")
}

// The innovation registry has to travel with the population, or a mutation
// after restoring could reissue a marking that is already in use.
func TestSnapshot_PreservesInnovations(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 60
	cfg.Seed = 5
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	pop, err = neat.Run(context.Background(), pop, xorEvaluator(), neat.RunOptions{MaxGenerations: 20})
	require.NoError(t, err)

	before := pop.Breeder.Innovations().Snapshot()
	require.NotEmpty(t, before.Connections, "a grown population should have issued connection markings")
	require.NotZero(t, before.CurrentID)

	encoded, err := json.Marshal(pop.Snapshot())
	require.NoError(t, err)
	var snapshot neat.Snapshot
	require.NoError(t, json.Unmarshal(encoded, &snapshot))
	restored, err := neat.Restore(snapshot)
	require.NoError(t, err)

	after := restored.Breeder.Innovations().Snapshot()
	assert.Equal(t, before.CurrentID, after.CurrentID)
	assert.ElementsMatch(t, before.Connections, after.Connections)
	assert.ElementsMatch(t, before.Splits, after.Splits)

	// The same structural change still resolves to the same marking.
	for _, connection := range before.Connections {
		assert.Equal(t, connection.ID,
			restored.Breeder.Innovations().ConnectionID(connection.From, connection.To))
	}
	// And a brand new one does not collide with anything already issued.
	fresh := restored.Breeder.Innovations().ConnectionID(-1, -2)
	assert.Greater(t, fresh, before.CurrentID)
}

func TestRestore_RejectsInconsistentSnapshots(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 10
	cfg.Seed = 1
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	pop = neat.Advance(pop)

	t.Run("no genomes", func(t *testing.T) {
		snapshot := pop.Snapshot()
		snapshot.Genomes = nil
		_, err := neat.Restore(snapshot)
		assert.ErrorContains(t, err, "no genomes")
	})

	t.Run("species pointing at a missing genome", func(t *testing.T) {
		snapshot := pop.Snapshot()
		require.NotEmpty(t, snapshot.Species)
		snapshot.Species[0].Genomes = []int{9999}
		_, err := neat.Restore(snapshot)
		assert.ErrorContains(t, err, "does not exist")
	})

	t.Run("invalid config", func(t *testing.T) {
		snapshot := pop.Snapshot()
		snapshot.Cfg.PopulationSize = 0
		_, err := neat.Restore(snapshot)
		assert.ErrorIs(t, err, neat.ErrInvalidConfig)
	})
}
