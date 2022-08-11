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
	cfg.PopulationSize = 128
	cfg.WeightReplaceRate = 0
	cfg.WeightMutationPower = 0.1
	cfg.MateBestRate = .5
	cfg.SpeciesStalenessThreshold = 30
	cfg.BiasNodes = 0
	cfg.HiddenActivationFns = []network.ActivationFunctionName{
		network.Sigmoid,
	}
	pop, err := neat.GeneratePopulation(cfg)
	if err != nil {
		panic(err)
	}

	for generation := 1; generation <= 1000; generation++ {
		pop = playGame(pop)
		bestFitness := pop.BestGenomeFitness

		fmt.Printf(`Generation %d
BestFitness: %f
-------------------------
`, generation, bestFitness)
		if bestFitness == 4 {
			fmt.Printf("Solved xor after %d generations with fitness %f\n", generation, bestFitness)
			runTest(pop.BestGenome)
			break
		}
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
					fitness += fitnessFunc(answer, output[0])
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
		fmt.Printf("input %v expect %v got %v\n", test.in, test.expected, output)
	}
}
