package neat_test

import (
	"github.com/jmwri/neatgo/neat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateDeleteConnection_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.DeleteConnectionMutationRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateDeleteConnection(cfg, genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateDeleteConnection_FullChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.DeleteConnectionMutationRate = 1
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateDeleteConnection(cfg, genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections()-1, actual.NumConnections())
}
