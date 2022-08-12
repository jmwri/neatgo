package main

import (
	"fmt"
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/network"
	"math"
	"math/rand"
	"sync"
	"time"
)

func main() {
	rand.Seed(time.Now().Unix())
	// 2 inputs, 1 output
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 150

	cfg.BiasNodes = 0

	cfg.OutputActivationFn = network.Sigmoid
	cfg.HiddenActivationFns = []network.ActivationFunctionName{
		network.Sigmoid,
	}

	cfg.AddNodeMutationRate = .5
	cfg.DeleteNodeMutationRate = .5
	cfg.BiasMutationRate = .8
	cfg.BiasMutationPower = .3
	cfg.BiasReplaceRate = .1

	cfg.AddConnectionMutationRate = .2
	cfg.WeightMutationRate = .8
	cfg.WeightMutationPower = .3
	cfg.WeightReplaceRate = .1

	cfg.SpeciesCompatExcessCoeff = 1
	cfg.SpeciesCompatBiasDiffCoeff = .5
	cfg.SpeciesCompatWeightDiffCoeff = .5
	cfg.SpeciesCompatThreshold = 2
	cfg.SpeciesStalenessThreshold = 20
	cfg.MateCrossoverRate = .75
	cfg.MateBestRate = .5

	pop, err := neat.GeneratePopulation(cfg)
	if err != nil {
		panic(err)
	}

	solved := false
	var generation int
	for generation = 1; generation <= 300; generation++ {
		pop = playGame(pop)
		bestFitness := pop.BestGenomeFitness

		fmt.Printf(`Generation %d
BestFitness: %f
-------------------------
`, generation, bestFitness)
		if bestFitness >= 3.9 {
			solved = true
			break
		}
	}
	if solved {
		fmt.Printf("Solved xor after %d generations with fitness %f\n", generation, pop.BestEverGenomeFitness)
	} else {
		fmt.Printf("Failed xor after %d generations with fitness %f\n", generation, pop.BestEverGenomeFitness)
	}

	runTest(pop.BestEverGenome)
	dumpGenome(pop.BestEverGenome)
}

func outputToAnswer(output []float64) float64 {
	if output[0] < .5 {
		return 0
	} else {
		return 1
	}
}

func playGame(pop neat.Population) neat.Population {
	clientStates := pop.States()
	wg := sync.WaitGroup{}
	wg.Add(len(clientStates))
	fitnessFunc := func(expectedOutput, output float64) float64 {
		return 1 - math.Pow(output-expectedOutput, 2)
	}
	for _, state := range clientStates {
		go func(state neat.ClientGenomeState) {
			defer wg.Done()
			inputs := [][]float64{
				{0, 0},
				{1, 0},
				{0, 1},
				{1, 1},
			}
			answer := [][]float64{
				{0},
				{1},
				{1},
				{0},
			}

			availableIndices := []int{0, 1, 2, 3}
			rand.Shuffle(len(availableIndices), func(i, j int) {
				availableIndices[i], availableIndices[j] = availableIndices[j], availableIndices[i]
			})

			fitness := .0
			for _, i := range availableIndices {
				input := inputs[i]
				answer := answer[i][0]
				state.SendInput() <- input
				select {
				case output := <-state.GetOutput():
					fitness += fitnessFunc(answer, outputToAnswer(output))
				case err := <-state.GetError():
					fmt.Printf("failed to process: %s\n", err)
				}
			}
			close(state.SendInput())

			state.SendFitness() <- fitness
			close(state.SendFitness())
		}(state)
	}
	pop = neat.RunGeneration(pop)
	wg.Wait()

	return pop
}

func runTest(genome neat.Genome) {
	type testCase struct {
		in, expected []float64
	}
	tests := []testCase{
		{
			in:       []float64{0, 0},
			expected: []float64{0},
		},
		{
			in:       []float64{1, 0},
			expected: []float64{1},
		},
		{
			in:       []float64{0, 1},
			expected: []float64{1},
		},
		{
			in:       []float64{1, 1},
			expected: []float64{0},
		},
	}
	for _, test := range tests {
		output, err := network.Activate(genome.Layers.Nodes(), genome.Connections, test.in)
		if err != nil {
			panic(err)
		}
		fmt.Printf("input %v expect %v got %v\n", test.in, test.expected, outputToAnswer(output))
	}
}

func dumpGenome(genome neat.Genome) {
	fmt.Println(genome.Layers)
	fmt.Println(genome.Connections)
}
