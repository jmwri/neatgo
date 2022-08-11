package neat_test

import (
	"fmt"
	"github.com/jmwri/neatgo/neat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateNodeBiases_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.BiasMutationRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateNodeBiases(cfg, genome)
	assert.Equal(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateNodeBiases_FullMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.BiasMutationRate = 1
	cfg.BiasReplaceRate = 1
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateNodeBiases(cfg, genome)
	assert.NotEqual(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateNodeBiases_MinimalMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.BiasMutationRate = 1
	cfg.BiasReplaceRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateNodeBiases(cfg, genome)
	assert.NotEqual(t, fmt.Sprint(genome), fmt.Sprint(actual))
}
