package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
	"math"
	"sync"
)

func GeneratePopulation(cfg Config) (Population, error) {
	genomes := make([]Genome, cfg.PopulationSize)
	genomeStates := make([]GenomeState, cfg.PopulationSize)
	pop := Population{
		Cfg:                   cfg,
		Genomes:               genomes,
		GenomeStates:          genomeStates,
		GenomeFitness:         make([]float64, cfg.PopulationSize),
		Species:               make([]Species, 0),
		Generation:            0,
		BestEverGenomeFitness: math.Inf(-1),
		BestGenomeFitness:     math.Inf(-1),
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
	GenomeStates          []GenomeState
	GenomeFitness         []float64
	Species               []Species
	Generation            int
	BestEverGenome        Genome
	BestEverGenomeFitness float64
	BestGenome            Genome
	BestGenomeFitness     float64
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
	pop = CullSpecies(pop)
	pop = FitnessSharing(pop)
	pop = KillStaleSpecies(pop)
	pop = KillBadSpecies(pop)
	pop = Evolve(pop)

	highestFitness := math.Inf(-1)
	for genomeID, fitness := range pop.GenomeFitness {
		if fitness > highestFitness {
			highestFitness = fitness
			pop.BestGenomeFitness = fitness
			pop.BestGenome = pop.Genomes[genomeID]
		}
	}
	if pop.BestGenomeFitness > pop.BestEverGenomeFitness {
		pop.BestEverGenomeFitness = pop.BestGenomeFitness
		pop.BestEverGenome = pop.BestGenome
	}

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
		output, err := network.Activate(genome.Layers.Nodes(), genome.Connections, input)
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

func Evolve(pop Population) Population {
	offspringCount := getDesiredOffspringCount(pop)
	newGenomes := make([]Genome, 0)
	newFitness := make([]float64, 0)
	newSpecies := make([]Species, 0)

	topSpeciesGenomes := make([]int, 0)

	for i, species := range pop.Species {
		// Take the top N genomes from each species for reproduction later
		for j := 0; j < pop.Cfg.TopGenomesFromSpeciesToFill; j++ {
			if j >= len(species.Genomes) {
				break
			}
			oldGenomeIndex := species.Genomes[j]
			topSpeciesGenomes = append(topSpeciesGenomes, oldGenomeIndex)
		}

		numOffspring, ok := offspringCount[i]
		if !ok {
			continue
		}
		if numOffspring < pop.Cfg.MinSpeciesSize {
			numOffspring = pop.Cfg.MinSpeciesSize
		}

		speciesGenomes := make([]int, 0)

		elitism := pop.Cfg.Elitism
		// Should only happen on the first run... Don't try to carry over more genomes than exist.
		if elitism > len(species.Genomes) {
			elitism = len(species.Genomes)
		}

		// Add elite genomes from each species with no mutation.
		for j := 0; j < elitism; j++ {
			oldGenomeIndex := species.Genomes[j]
			newGenomeIndex := len(newGenomes)
			newGenomes = append(newGenomes, pop.Genomes[oldGenomeIndex])
			newFitness = append(newFitness, pop.GenomeFitness[oldGenomeIndex])
			speciesGenomes = append(speciesGenomes, newGenomeIndex)
		}

		// Fill the remaining allowance with mutated offspring.
		for j := elitism; j < numOffspring; j++ {
			if len(pop.Species[i].Genomes) == 0 {
				continue
			}
			// Get offspring from the species
			offspring := GetOffspring(pop, pop.Species[i])
			newGenomeIndex := len(newGenomes)
			newGenomes = append(newGenomes, offspring)
			newFitness = append(newFitness, 0)
			speciesGenomes = append(speciesGenomes, newGenomeIndex)
		}
		species.Genomes = speciesGenomes
		newSpecies = append(newSpecies, species)
	}

	// Once processed all species, if we have some population size left over, crossover the best genomes of all species.
	// Don't add these genomes to any species.
	if len(newGenomes) < pop.Cfg.PopulationSize && len(topSpeciesGenomes) != 0 {
		for len(newGenomes) < pop.Cfg.PopulationSize {
			a := util.RandSliceElement(topSpeciesGenomes)
			b := util.RandSliceElement(topSpeciesGenomes)
			if pop.GenomeFitness[a] < pop.GenomeFitness[b] {
				a, b = b, a
			}
			aGenome := pop.Genomes[a]
			bGenome := pop.Genomes[b]
			baby := Crossover(pop.Cfg, aGenome, bGenome)
			newGenomes = append(newGenomes, baby)
			newFitness = append(newFitness, 0)
		}
	}

	pop.Genomes = newGenomes
	pop.GenomeFitness = newFitness
	pop.Species = newSpecies
	pop.GenomeStates = make([]GenomeState, len(newGenomes))
	return pop
}
