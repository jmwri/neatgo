package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/network"
	"sync"
)

func GeneratePopulation(cfg Config) (Population, error) {
	genomes := make([]Genome, cfg.PopulationSize)
	genomeStates := make([]GenomeState, cfg.PopulationSize)
	pop := Population{
		Cfg:           cfg,
		Genomes:       genomes,
		GenomeStates:  genomeStates,
		GenomeFitness: make([]float64, cfg.PopulationSize),
		Species:       make([]Species, 0),
		Generation:    0,
	}
	var err error
	for i := 0; i < cfg.PopulationSize; i++ {
		genomes[i], err = GenerateGenome(cfg)
		if err != nil {
			return pop, fmt.Errorf("failed to generate genome: %w", err)
		}
	}
	return buildGenomeStates(pop), nil
}

type Population struct {
	Cfg     Config
	Genomes []Genome
	// GenomeStates contains a GenomeState and should be used for the Genome as the same index.
	GenomeStates  []GenomeState
	GenomeFitness []float64
	Species       []Species
	Generation    int
}

func (p Population) States() []ClientGenomeState {
	clientStates := make([]ClientGenomeState, len(p.GenomeStates))
	for i := range p.GenomeStates {
		clientStates[i] = p.GenomeStates[i]
	}
	return clientStates
}

type ClientGenomeState interface {
	// SendInput returns a channel where the client can send input to be processed through the network.
	SendInput() chan<- []float64
	// SendFitness returns a channel where the client can send the fitness of the genome.
	SendFitness() chan<- float64
	// GetOutput returns a channel where the client can receive the output from the network.
	GetOutput() <-chan []float64
	// GetError returns a channel where errors can be received.
	GetError() <-chan error
}
type BackendGenomeState interface {
	// GetInput returns a channel where the backend can receive input to be processed through the network.
	GetInput() <-chan []float64
	// GetFitness returns a channel where the backend can receive the fitness of the genome.
	GetFitness() <-chan float64
	// SendOutput returns a channel where the backend can send the output from the network.
	SendOutput() chan<- []float64
	// SendError returns a channel where the backend can send any errors.
	SendError() chan<- error
}

type genomeState struct {
	inputCh   chan []float64
	fitnessCh chan float64
	outputCh  chan []float64
	errCh     chan error
}

func (s genomeState) SendInput() chan<- []float64 {
	return s.inputCh
}

func (s genomeState) GetInput() <-chan []float64 {
	return s.inputCh
}

func (s genomeState) SendFitness() chan<- float64 {
	return s.fitnessCh
}

func (s genomeState) GetFitness() <-chan float64 {
	return s.fitnessCh
}

func (s genomeState) SendOutput() chan<- []float64 {
	return s.outputCh
}

func (s genomeState) GetOutput() <-chan []float64 {
	return s.outputCh
}

func (s genomeState) SendError() chan<- error {
	return s.errCh
}

func (s genomeState) GetError() <-chan error {
	return s.errCh
}

type GenomeState interface {
	ClientGenomeState
	BackendGenomeState
}

func RunGeneration(pop Population) Population {
	pop.Generation++
	wg := sync.WaitGroup{}
	wg.Add(len(pop.Genomes))
	for i := range pop.Genomes {
		go runGenome(&wg, pop, i)
	}

	// Wait for all genomes in population to finish.
	wg.Wait()

	pop = Speciate(pop)
	pop = RankSpecies(pop)
	// TODO: Rank species - sort species by their average fitness
	// TODO: Cull species - remove the bottom 50% of each species
	// TODO: Kill stale species - remove species that haven't improved in the past N generations
	// TODO: Kill unreproducable species - remove species that won't be able to reproduce (based on species target size)

	// TODO: Mate genomes to fill the rest of the population
	// Species size should be calculated by their performance against all others

	// Build fresh genome states for next generation.
	return buildGenomeStates(pop)
}

func runGenome(wg *sync.WaitGroup, pop Population, i int) {
	genome := pop.Genomes[i]
	var state BackendGenomeState = pop.GenomeStates[i]
	defer wg.Done()
	defer close(state.SendOutput())
	defer close(state.SendError())
	for {
		input, ok := <-state.GetInput()
		if !ok {
			// If input is closed, then game has finished.
			fitness, ok := <-state.GetFitness()
			if !ok {
				state.SendError() <- fmt.Errorf("failed to receive fitness")
				return
			}
			pop.GenomeFitness[i] = fitness
			return
		}
		output, err := network.Activate(genome.layers.Nodes(), genome.connections, input)
		if err != nil {
			state.SendError() <- err
			continue
		}
		state.SendOutput() <- output
	}
}

func buildGenomeStates(pop Population) Population {
	for i := range pop.GenomeStates {
		pop.GenomeStates[i] = genomeState{
			inputCh:   make(chan []float64),
			fitnessCh: make(chan float64),
			outputCh:  make(chan []float64),
			errCh:     make(chan error),
		}
	}
	return pop
}
