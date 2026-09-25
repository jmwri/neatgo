package gpu_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jmwri/neatgo/v2/gpu"
	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
)

func device(tb testing.TB) *gpu.Device {
	tb.Helper()
	if !gpu.Available() {
		tb.Skip("no CUDA device")
	}
	d, err := gpu.Open(0)
	require.NoError(tb, err)
	tb.Cleanup(func() { d.Close() })
	return d
}

var xorInputs = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
var xorAnswers = []float64{0, 1, 1, 0}

func xorScore(_ *network.Network, out network.Outputs) (float64, error) {
	fitness := 0.0
	for i := 0; i < out.Len(); i++ {
		fitness += 1 - math.Pow(out.Sample(i)[0]-xorAnswers[i], 2)
	}
	return fitness, nil
}

// The GPU is a drop-in Activator: a whole NEAT run on it finds a network that
// solves XOR, and that network, run on the CPU in float64, still solves it.
func TestRunBatchOnTheGPUSolvesXOR(t *testing.T) {
	d := device(t)
	cfg := neat.DefaultConfig(2, 1)
	cfg.Seed = 1
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	pop, err = neat.RunBatch(context.Background(), pop, neat.Batch{Activator: d, Inputs: xorInputs, Score: xorScore},
		neat.RunOptions{MaxGenerations: 300, Solved: func(p neat.Population) bool { return p.BestGenomeFitness >= 3.9 }})
	require.NoError(t, err)
	require.GreaterOrEqual(t, pop.BestEverGenomeFitness, 3.9)

	best, err := pop.BestEverGenome.Compile()
	require.NoError(t, err)
	for i, input := range xorInputs {
		out, err := best.Activate(input)
		require.NoError(t, err)
		assert.Equal(t, xorAnswers[i], math.Round(out[0]), "input %v", input)
	}
}

// evolved returns a population that has grown some structure, which is what a
// real run hands the activator.
func evolved(tb testing.TB, size int) []*network.Network {
	cfg := neat.DefaultConfig(8, 2)
	cfg.PopulationSize = size
	cfg.Seed = 3
	cfg.AddNodeMutationRate, cfg.AddConnectionMutationRate = .3, .5
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(tb, err)
	rng := rand.New(rand.NewPCG(1, 1))
	for g := 0; g < 25; g++ {
		for i := range pop.GenomeFitness {
			pop.GenomeFitness[i] = rng.Float64()
		}
		pop = neat.Advance(pop)
	}
	nets := make([]*network.Network, len(pop.Genomes))
	for i, g := range pop.Genomes {
		nets[i], err = g.Compile()
		require.NoError(tb, err)
	}
	return nets
}

// BenchmarkActivateBatch shows where the GPU starts to pay: a population of 150
// over datasets of growing size, against the CPU using every core.
func BenchmarkActivateBatch(b *testing.B) {
	d := device(b)
	nets := evolved(b, 150)
	nodes := 0
	for _, n := range nets {
		nodes += n.NumNodes()
	}
	b.Logf("%d networks, %.1f nodes each on average", len(nets), float64(nodes)/float64(len(nets)))

	rng := rand.New(rand.NewPCG(2, 2))
	for _, samples := range []int{4, 100, 1000, 10000, 100000} {
		inputs := make([][]float64, samples)
		for i := range inputs {
			inputs[i] = make([]float64, 8)
			for j := range inputs[i] {
				inputs[i][j] = rng.Float64()
			}
		}
		b.Run(fmt.Sprintf("cpu/samples=%d", samples), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := network.ActivateBatch(nets, inputs, 0); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("gpu/samples=%d", samples), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := d.ActivateBatch(nets, inputs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
