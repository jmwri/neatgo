package neat_test

import (
	"context"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// delaySequence is a fixed bit sequence; the task is to output, at each step,
// the bit that was shown one step earlier. No feed-forward network can do this
// - the answer is not a function of the current input - so solving it proves
// that mutation grows a loop, that the loop compiles, and that Memory carries
// the right value between steps.
var delaySequence = []float64{0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0}

func delayFitness(_ context.Context, net *network.Network) (float64, error) {
	mem := net.NewMemory()
	output := make([]float64, net.NumOutputs())
	fitness := .0
	for i, bit := range delaySequence {
		if err := net.Step(mem, []float64{bit}, output); err != nil {
			return 0, err
		}
		if i == 0 {
			continue
		}
		diff := output[0] - delaySequence[i-1]
		fitness += 1 - diff*diff
	}
	return fitness, nil
}

// End to end for the recurrent option: the whole pipeline has to be able to
// learn something that depends on the previous input.
func TestEvolution_LearnsToRememberPreviousInput(t *testing.T) {
	if testing.Short() {
		t.Skip("evolutionary run")
	}

	cfg := neat.DefaultConfig(1, 1)
	cfg.PopulationSize = 150
	cfg.Recurrent = true
	cfg.Seed = 5
	cfg.HiddenActivationFns = []network.ActivationFunctionName{network.Sigmoid}

	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	maximum := float64(len(delaySequence) - 1)
	pop, err = neat.Run(context.Background(), pop, delayFitness, neat.RunOptions{
		MaxGenerations: 300,
		Solved:         func(pop neat.Population) bool { return pop.BestEverGenomeFitness >= .95*maximum },
	})
	require.NoError(t, err)
	assert.GreaterOrEqual(t, pop.BestEverGenomeFitness, .95*maximum, "failed to learn a one step delay")

	net, err := pop.BestEverGenome.CompileRecurrent()
	require.NoError(t, err)
	assert.True(t, hasBackwardConnection(pop.BestEverGenome), "the solution has to loop")

	// Fresh memory, and it reproduces the sequence one step late.
	mem := net.NewMemory()
	output := make([]float64, 1)
	for i, bit := range delaySequence {
		require.NoError(t, net.Step(mem, []float64{bit}, output))
		if i > 0 {
			assert.InDelta(t, delaySequence[i-1], output[0], .5, "step %d", i)
		}
	}
}
