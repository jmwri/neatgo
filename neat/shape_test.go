package neat_test

import (
	"context"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A fitness slice that is not one entry per genome is an error, not a panic on
// a worker goroutine that no caller can recover from.
func TestEvaluate_RejectsMismatchedFitnessSlice(t *testing.T) {
	pop := testPopulation(t, 10)
	pop.GenomeFitness = pop.GenomeFitness[:3]

	err := neat.Evaluate(context.Background(), pop, constantFitness(1))
	assert.ErrorIs(t, err, neat.ErrPopulationShape)
}

// A snapshot whose ID counter has fallen behind the markings its genomes hold
// - hand edited, or merged from two runs - must not reissue any of them.
func TestRestore_MovesIDCounterPastEveryMarking(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 10
	cfg.Seed = 4
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	pop = neat.Advance(pop)

	snapshot := pop.Snapshot()
	snapshot.Innovations.CurrentID = 0
	snapshot.Genomes[0].Connections[0].ID = 5000

	restored, err := neat.Restore(snapshot)
	require.NoError(t, err)
	fresh := restored.Breeder.Innovations().ConnectionID(-1, -2)
	assert.Greater(t, fresh, 5000)
}
