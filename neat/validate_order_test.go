package neat_test

import (
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The problems a config has must be reported in the same order every time, so
// a message can be compared in a test or a log rather than shuffled by map
// iteration.
func TestValidate_ReportsProblemsInAStableOrder(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.AddNodeMutationRate = 2
	cfg.MateBestRate = -1
	cfg.WeightInitStdDev = -1
	cfg.BiasMutationPower = -1
	cfg.OutputActivationFn = "nope"
	cfg.InputActivationFn = "nope"

	first := cfg.Validate()
	require.Error(t, first)
	for i := 0; i < 20; i++ {
		assert.Equal(t, first.Error(), cfg.Validate().Error())
	}
}
