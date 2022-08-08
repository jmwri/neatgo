package main

import (
	"fmt"
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/network"
	"math"
	"sync"
)

func main() {
	// 2 inputs, 1 output
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 150
	pop, err := neat.GeneratePopulation(cfg)
	if err != nil {
		panic(err)
	}

	for generation := 1; generation <= 300; generation++ {
		pop = playGame(pop)
		bestFitness := pop.GenomeFitness[pop.BestGenome]

		fmt.Printf(`
Generation %d
BestFitness: %f
-------------------------
`, generation, bestFitness)
	}

	best := pop.Genomes[pop.BestGenome]

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
		output, err := network.Activate(best.Layers.Nodes(), best.Connections, test.in)
		if err != nil {
			panic(err)
		}
		fmt.Printf("input %v expect %v got %v\n", test.in, test.expected, output)
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
			fitness := .0

			state.SendInput() <- []float64{0, 0}
			select {
			case output := <-state.GetOutput():
				fitness += fitnessFunc(1, output[0])
			case err := <-state.GetError():
				fmt.Printf("failed to process: %s\n", err)
			}

			state.SendInput() <- []float64{1, 0}
			select {
			case output := <-state.GetOutput():
				fitness += fitnessFunc(0, output[0])
			case err := <-state.GetError():
				fmt.Printf("failed to process: %s\n", err)
			}

			state.SendInput() <- []float64{0, 1}
			select {
			case output := <-state.GetOutput():
				fitness += fitnessFunc(0, output[0])
			case err := <-state.GetError():
				fmt.Printf("failed to process: %s\n", err)
			}

			state.SendInput() <- []float64{1, 1}
			select {
			case output := <-state.GetOutput():
				fitness += fitnessFunc(1, output[0])
			case err := <-state.GetError():
				fmt.Printf("failed to process: %s\n", err)
			}

			close(state.SendInput())

			fitness = math.Pow(math.Max(fitness*100-200, 1), 2)

			state.SendFitness() <- fitness
			close(state.SendFitness())
		}(state)
	}
	pop = neat.RunGeneration(pop)
	wg.Wait()

	return pop
}
