package main

import (
	"fmt"
	"github.com/jmwri/neatgo/neat"
	"math"
	"sort"
	"sync"
)

func main() {
	// 3 inputs, 1 output
	cfg := neat.DefaultConfig(3, 1)
	cfg.PopulationSize = 100
	pop, err := neat.GeneratePopulation(cfg)
	if err != nil {
		panic(err)
	}

	for generation := 1; generation <= 300; generation++ {
		pop = playGame(pop)
		speciesBestFitnesses := make([]float64, len(pop.Species))
		speciesAvgFitnesses := make([]float64, len(pop.Species))
		for i, species := range pop.Species {
			speciesBestFitnesses[i] = species.BestFitness
			speciesAvgFitnesses[i] = species.AvgFitness
		}

		geneCounts := make(map[int]int)
		for _, genome := range pop.Genomes {
			count := genome.NumNodes() + genome.NumConnections()
			geneCounts[count] += 1
		}

		sort.Slice(speciesBestFitnesses, func(i, j int) bool {
			return speciesBestFitnesses[i] > speciesBestFitnesses[j]
		})
		sort.Slice(speciesAvgFitnesses, func(i, j int) bool {
			return speciesAvgFitnesses[i] > speciesAvgFitnesses[j]
		})
		fmt.Printf(`
Generation %d
NumSpecies: %d
AvgFitnesses: %v
BestFitnesses: %v
-------------------------
`, generation, len(pop.Species), speciesAvgFitnesses, speciesBestFitnesses)
	}
}

func playGame(pop neat.Population) neat.Population {
	clientStates := pop.States()
	wg := sync.WaitGroup{}
	wg.Add(len(clientStates))
	for _, state := range clientStates {
		go func(state neat.ClientGenomeState) {
			defer wg.Done()
			fitness := 0.0

			state.SendInput() <- []float64{0, 0, 0}
			select {
			case output := <-state.GetOutput():
				fitness += 1 - output[0]
			case err := <-state.GetError():
				fmt.Printf("failed to process: %s\n", err)
			}

			state.SendInput() <- []float64{1, 0, 1}
			select {
			case output := <-state.GetOutput():
				fitness += output[0]
			case err := <-state.GetError():
				fmt.Printf("failed to process: %s\n", err)
			}

			state.SendInput() <- []float64{0, 1, 1}
			select {
			case output := <-state.GetOutput():
				fitness += output[0]
			case err := <-state.GetError():
				fmt.Printf("failed to process: %s\n", err)
			}

			state.SendInput() <- []float64{1, 1, 1}
			select {
			case output := <-state.GetOutput():
				fitness += 1 - output[0]
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
