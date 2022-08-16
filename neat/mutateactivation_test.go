package neat_test

import (
	"fmt"
	"github.com/jmwri/neatgo/neat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateNodeActivations_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.ActivationMutationRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateNodeActivations(cfg, genome)
	assert.Equal(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateNodeActivations_FullMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.ActivationMutationRate = 1
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateNodeActivations(cfg, genome)
	assert.NotEqual(t, fmt.Sprint(genome), fmt.Sprint(actual))
}
