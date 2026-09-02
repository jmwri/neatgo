package neat_test

import (
	"context"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A snapshot held in memory while the run continues must keep describing the
// generation it was taken at. The fitness slice is the one thing the next
// evaluation writes into in place, so it has to be copied.
func TestSnapshot_IsNotAViewOfTheLivePopulation(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 12
	cfg.Seed = 9
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	pop, err = neat.RunGeneration(context.Background(), pop, constantFitness(1))
	require.NoError(t, err)

	snapshot := pop.Snapshot()
	before := append([]float64{}, snapshot.GenomeFitness...)

	_, err = neat.RunGeneration(context.Background(), pop, constantFitness(42))
	require.NoError(t, err)

	assert.Equal(t, before, snapshot.GenomeFitness, "the next evaluation must not write into the snapshot")
}
