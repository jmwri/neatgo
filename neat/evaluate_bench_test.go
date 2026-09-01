package neat_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
)

// benchEvaluator does a fixed amount of CPU work per genome, standing in for a
// real fitness function.
func benchEvaluator(activations int) neat.Evaluator {
	return func(_ context.Context, net *network.Network) (float64, error) {
		input := make([]float64, net.NumInputs())
		output := make([]float64, net.NumOutputs())
		fitness := .0
		for i := 0; i < activations; i++ {
			for j := range input {
				input[j] = math.Sin(float64(i + j))
			}
			if err := net.ActivateInto(input, output); err != nil {
				return 0, err
			}
			fitness += output[0]
		}
		return fitness, nil
	}
}

// BenchmarkEvaluate shows how a generation's evaluation scales with the number
// of workers. Parallelism 1 is the sequential baseline.
func BenchmarkEvaluate(b *testing.B) {
	cfg := neat.DefaultConfig(8, 4)
	cfg.PopulationSize = 150
	pop, err := neat.GeneratePopulation(cfg)
	if err != nil {
		b.Fatal(err)
	}

	eval := benchEvaluator(200)
	ctx := context.Background()

	for _, parallelism := range []int{1, 2, 4, 8, 16, neat.Unlimited} {
		name := fmt.Sprintf("workers=%d", parallelism)
		if parallelism == neat.Unlimited {
			name = "workers=unlimited"
		}
		pop.Cfg.Parallelism = parallelism
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if err := neat.Evaluate(ctx, pop, eval); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
