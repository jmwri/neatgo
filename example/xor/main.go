package main

import (
	"context"
	"fmt"
	"log"
	"math"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
)

// solvedFitness is the fitness at which every one of the four cases is
// answered correctly with a comfortable margin. The maximum is 4.
const solvedFitness = 3.9

var xorInputs = [][]float64{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1},
}

var xorAnswers = []float64{0, 1, 1, 0}

// evaluate scores one genome. It is called concurrently for the whole
// population, once per genome per generation.
func evaluate(_ context.Context, net *network.Network) (float64, error) {
	// Reused across all four cases so the hot loop allocates nothing.
	output := make([]float64, net.NumOutputs())

	fitness := .0
	for i, input := range xorInputs {
		if err := net.ActivateInto(input, output); err != nil {
			return 0, err
		}
		// Squared error, so a confident right answer scores better than a
		// hesitant one.
		fitness += 1 - math.Pow(output[0]-xorAnswers[i], 2)
	}
	return fitness, nil
}

func main() {
	// 2 inputs, 1 output. NEAT starts minimal and grows the hidden structure
	// itself, so no hidden layer is specified here.
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 150

	// Output nodes carry their own bias, so no separate bias node is needed.
	cfg.BiasNodes = 0
	cfg.OutputActivationFn = network.Sigmoid
	cfg.HiddenActivationFns = []network.ActivationFunctionName{
		network.Sigmoid,
	}

	// Leaving cfg.Seed at zero draws a seed and records it on the population,
	// so a run worth keeping can be replayed by setting cfg.Seed to it.
	pop, err := neat.GeneratePopulation(cfg)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("seed %d\n", pop.Seed)

	pop, err = neat.Run(context.Background(), pop, evaluate, neat.RunOptions{
		MaxGenerations: 300,
		Solved: func(pop neat.Population) bool {
			return pop.BestGenomeFitness >= solvedFitness
		},
		OnGeneration: func(pop neat.Population) error {
			fmt.Printf("Generation %3d | best %.4f | nodes %2d | connections %2d | pop %3d | species %2d\n",
				pop.Generation,
				pop.BestGenomeFitness,
				pop.BestGenome.NumNodes(),
				pop.BestGenome.NumConnections(),
				len(pop.Genomes),
				len(pop.Species),
			)
			return nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	if pop.BestEverGenomeFitness >= solvedFitness {
		fmt.Printf("\nSolved xor after %d generations with fitness %f\n", pop.Generation, pop.BestEverGenomeFitness)
	} else {
		fmt.Printf("\nFailed xor after %d generations with fitness %f\n", pop.Generation, pop.BestEverGenomeFitness)
	}

	best, err := pop.BestEverGenome.Compile()
	if err != nil {
		log.Fatal(err)
	}
	runTest(best)
	dumpGenome(pop.BestEverGenome)
}

func runTest(net *network.Network) {
	correct := 0
	for i, input := range xorInputs {
		output, err := net.Activate(input)
		if err != nil {
			log.Fatal(err)
		}
		answer := math.Round(output[0])
		if answer == xorAnswers[i] {
			correct++
		}
		fmt.Printf("input %v expect %v got %v (raw %.4f)\n", input, xorAnswers[i], answer, output[0])
	}
	fmt.Printf("%d/%d correct\n", correct, len(xorInputs))
}

func dumpGenome(genome neat.Genome) {
	for i, layer := range genome.Layers {
		fmt.Printf("layer %d: %v\n", i, layer)
	}
	for _, connection := range genome.Connections {
		if !connection.Enabled {
			continue
		}
		fmt.Printf("connection %d: %d -> %d weight %.4f\n", connection.ID, connection.From, connection.To, connection.Weight)
	}
}
