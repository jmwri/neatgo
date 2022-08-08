package neat_test

import (
	"fmt"
	"github.com/jmwri/neatgo/neat"
	"github.com/stretchr/testify/assert"
	"math"
	"sort"
	"sync"
	"testing"
)

func TestRunGeneration(t *testing.T) {
	cfg := neat.DefaultConfig(1, 5)
	pop, err := neat.GeneratePopulation(cfg)
	assert.NoError(t, err)

	pop = playGame(pop)
}

func playGame(pop neat.Population) neat.Population {
	for pop.Generation < 100 {
		clientStates := pop.States()
		wg := sync.WaitGroup{}
		wg.Add(len(clientStates))
		for _, state := range clientStates {
			go func(state neat.ClientGenomeState) {
				fitness := 0.0
				defer wg.Done()
				defer close(state.SendFitness())
				defer func() {
					state.SendFitness() <- fitness
				}()
				defer close(state.SendInput())

				// Silly test game.
				// Go from 1 to 5, output should ideally match the input
				for i := 1; i <= 5; i++ {
					iFloat := float64(i)
					state.SendInput() <- []float64{iFloat}
					output := <-state.GetOutput()
					var bestGuess int
					bestGuessScore := math.Inf(-1)
					for guessIndex, score := range output {
						if score > bestGuessScore {
							bestGuess = guessIndex + 1
							bestGuessScore = score
						}
					}

					bestGuessFloat := float64(bestGuess)
					var distanceFromCorrect float64
					if bestGuessFloat < iFloat {
						distanceFromCorrect = iFloat - bestGuessFloat
					} else {
						distanceFromCorrect = bestGuessFloat - iFloat
					}
					fitness += iFloat - distanceFromCorrect
				}
			}(state)
		}
		pop = neat.RunGeneration(pop)
		wg.Wait()

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

		fmt.Printf("Generation %d\nSpecies fitnesses: %v\nSpecies avg fitnesses: %v\nGene counts: %v\n", pop.Generation, speciesBestFitnesses, speciesAvgFitnesses, geneCounts)
	}
	return pop
}
