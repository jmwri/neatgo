package neat

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"sync"
	"testing"
)

func TestRunGeneration(t *testing.T) {
	cfg := DefaultConfig(1, 5)
	cfg.PopulationSize = 5
	pop, err := GeneratePopulation(cfg)
	assert.NoError(t, err)

	clientStates := pop.States()
	wg := sync.WaitGroup{}
	wg.Add(len(clientStates))
	for _, state := range clientStates {
		go func(state ClientGenomeState) {
			fitness := 0.0
			defer wg.Done()
			defer close(state.SendFitness())
			defer func() {
				state.SendFitness() <- fitness
			}()
			defer close(state.SendInput())

			// Silly test game.
			// Go from 1 to 5, output should ideally match the input
			for i := 1.0; i <= 5; i++ {
				state.SendInput() <- []float64{i}
				output := <-state.GetOutput()
				var bestGuess int
				var bestGuessScore float64
				for guessIndex, score := range output {
					if score > bestGuessScore {
						bestGuess = guessIndex + 1
						bestGuessScore = score
					}
				}
				fitness += i - (i - float64(bestGuess))
			}
		}(state)
	}
	RunGeneration(pop)
	wg.Wait()
	fmt.Println("all done")
}
