package neat_test

import (
	"github.com/jmwri/neatgo/v2/neat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateDeleteConnection_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.DeleteConnectionMutationRate = 0
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateDeleteConnection(genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateDeleteConnection_FullChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.DeleteConnectionMutationRate = 1
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateDeleteConnection(genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections()-1, actual.NumConnections())
}
