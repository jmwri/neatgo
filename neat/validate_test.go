package neat_test

import (
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfig_ValidateAcceptsDefaults(t *testing.T) {
	assert.NoError(t, neat.DefaultConfig(2, 1).Validate())
	assert.NoError(t, neat.DefaultConfig(4, 6, 2).Validate())
}

func TestConfig_ValidateRejectsBadSettings(t *testing.T) {
	tests := []struct {
		name   string
		break_ func(*neat.Config)
		want   string
	}{
		{"no output layer", func(c *neat.Config) { c.Layers = []int{2} }, "at least an input and an output"},
		{"empty layer", func(c *neat.Config) { c.Layers = []int{2, 0, 1} }, "Layers[1]"},
		{"tiny population", func(c *neat.Config) { c.PopulationSize = 1 }, "PopulationSize"},
		{"rate above one", func(c *neat.Config) { c.AddNodeMutationRate = 1.5 }, "AddNodeMutationRate"},
		{"negative rate", func(c *neat.Config) { c.WeightMutationRate = -.1 }, "WeightMutationRate"},
		{"inverted weight range", func(c *neat.Config) { c.MinWeight, c.MaxWeight = 5, -5 }, "MinWeight"},
		{"survival threshold above one", func(c *neat.Config) { c.SurvivalThreshold = 2 }, "SurvivalThreshold"},
		{"zero survival threshold", func(c *neat.Config) { c.SurvivalThreshold = 0 }, "SurvivalThreshold"},
		{"negative compat threshold", func(c *neat.Config) { c.SpeciesCompatThreshold = -1 }, "SpeciesCompatThreshold"},
		{"zero staleness", func(c *neat.Config) { c.SpeciesStalenessThreshold = 0 }, "SpeciesStalenessThreshold"},
		{"zero min species size", func(c *neat.Config) { c.MinSpeciesSize = 0 }, "MinSpeciesSize"},
		{"elitism beyond population", func(c *neat.Config) { c.Elitism = 500 }, "Elitism"},
		{"bad parallelism", func(c *neat.Config) { c.Parallelism = -7 }, "Parallelism"},
		{"unknown output activation", func(c *neat.Config) { c.OutputActivationFn = "nope" }, "OutputActivationFn"},
		{"no hidden activations", func(c *neat.Config) { c.HiddenActivationFns = nil }, "HiddenActivationFns"},
		{"unknown hidden activation", func(c *neat.Config) {
			c.HiddenActivationFns = []network.ActivationFunctionName{"nope"}
		}, "HiddenActivationFns[0]"},
		{"negative mutation power", func(c *neat.Config) { c.WeightMutationPower = -1 }, "WeightMutationPower"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := neat.DefaultConfig(2, 1)
			test.break_(&cfg)
			err := cfg.Validate()
			require.Error(t, err)
			assert.ErrorContains(t, err, test.want)

			// The same problem must stop a population being built, rather than
			// surfacing much later as odd behaviour.
			_, err = neat.GeneratePopulation(cfg)
			assert.ErrorContains(t, err, test.want)
		})
	}
}

// All the problems are reported at once, so a bad config can be fixed in one
// pass rather than one error at a time.
func TestConfig_ValidateReportsEveryProblem(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 0
	cfg.SurvivalThreshold = 3
	cfg.MateBestRate = 9

	err := cfg.Validate()
	require.Error(t, err)
	assert.ErrorContains(t, err, "PopulationSize")
	assert.ErrorContains(t, err, "SurvivalThreshold")
	assert.ErrorContains(t, err, "MateBestRate")
}
