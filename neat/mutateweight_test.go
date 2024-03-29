package neat_test

import (
	"fmt"
	"github.com/jmwri/neatgo/neat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateConnectionWeights_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.WeightMutationRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateConnectionWeights(cfg, genome)
	assert.Equal(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateConnectionWeights_FullMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.WeightMutationRate = 1
	cfg.WeightReplaceRate = 1
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateConnectionWeights(cfg, genome)
	assert.NotEqual(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateConnectionWeights_MinimalMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.WeightMutationRate = 1
	cfg.WeightReplaceRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateConnectionWeights(cfg, genome)
	assert.NotEqual(t, fmt.Sprint(genome), fmt.Sprint(actual))
}
