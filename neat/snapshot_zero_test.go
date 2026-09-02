package neat_test

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A checkpoint taken before the first generation has no best genome yet, which
// the population records as a fitness of -Inf. encoding/json cannot write that,
// so a run that checkpoints before it starts must not fail on the spot.
func TestSnapshot_BeforeFirstGenerationRoundTrips(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 20
	cfg.Seed = 11
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	require.True(t, math.IsInf(pop.BestGenomeFitness, -1))

	encoded, err := json.Marshal(pop.Snapshot())
	require.NoError(t, err, "a snapshot of a fresh population must serialise")

	var snapshot neat.Snapshot
	require.NoError(t, json.Unmarshal(encoded, &snapshot))
	restored, err := neat.Restore(snapshot)
	require.NoError(t, err)

	assert.True(t, math.IsInf(restored.BestGenomeFitness, -1), "no best genome yet must restore as -Inf")
	assert.True(t, math.IsInf(restored.BestEverGenomeFitness, -1))

	// And the restored run is the same run.
	eval := xorEvaluator()
	fresh, err := neat.Run(context.Background(), pop, eval, neat.RunOptions{MaxGenerations: 8})
	require.NoError(t, err)
	resumed, err := neat.Run(context.Background(), restored, eval, neat.RunOptions{MaxGenerations: 8})
	require.NoError(t, err)
	assert.Equal(t, fingerprint(fresh), fingerprint(resumed))
}

// Two snapshots of the same registry must serialise identically, so that a
// checkpoint can be compared or deduplicated by its bytes.
func TestInnovationsSnapshot_IsDeterministic(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 40
	cfg.Seed = 3
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	pop, err = neat.Run(context.Background(), pop, xorEvaluator(), neat.RunOptions{MaxGenerations: 15})
	require.NoError(t, err)

	first := pop.Breeder.Innovations().Snapshot()
	require.Greater(t, len(first.Connections), 1)
	for i := 0; i < 10; i++ {
		assert.Equal(t, first, pop.Breeder.Innovations().Snapshot())
	}
}
