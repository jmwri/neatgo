package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/aggregation"
)

func NewPopulation(cfg *Config, reproduction *Reproduction) (*Population, error) {
	initPopulation, err := reproduction.CreateNew(cfg.PopulationSize)
	species := NewSpeciesSet(cfg)
	species.Speciate(cfg, initPopulation, 0)
	if err != nil {
		return nil, err
	}
	return &Population{
		cfg:              cfg,
		reproduction:     reproduction,
		fitnessCriterion: cfg.FitnessCriterion,
		population:       initPopulation,
		species:          species,
		generation:       0,
		bestGenome:       nil,
	}, nil
}

type Population struct {
	cfg              *Config
	reproduction     *Reproduction
	fitnessCriterion aggregation.Fn
	population       map[int64]*Genome
	species          *SpeciesSet
	generation       int
	bestGenome       *Genome
}

func (p *Population) Run(fitnessFn FitnessFn, generations int) error {
	if p.cfg.NoFitnessTermination && generations == 0 {
		return fmt.Errorf("cannot have no generational limit with no fitness termination")
	}

	generation := 0
	for {
		if generations != 0 && generation >= generations {
			break
		}

		var genBestGenome *Genome
		fitnesses := make([]float64, 0)
		for _, genome := range p.population {
			fitness := fitnessFn(genome, generation)
			genome.fitness = fitness
			fitnesses = append(fitnesses, fitness)

			if genBestGenome == nil || fitness > genBestGenome.fitness {
				genBestGenome = genome
			}
		}

		isBestGenomeEver := p.bestGenome == nil || (genBestGenome != nil && genBestGenome.fitness > p.bestGenome.fitness)

		if isBestGenomeEver {
			p.bestGenome = genBestGenome
		}

		if !p.cfg.NoFitnessTermination {
			// End if fitness termination is reached

			aggFitness := p.fitnessCriterion(fitnesses)
			if aggFitness >= p.cfg.FitnessThreshold {
				// Found solution
				break
			}
		}

		p.population = p.reproduction.Reproduce(p.cfg, p.species, p.cfg.PopulationSize, generation)

		if len(p.species.species) == 0 {
			// Extinct!
			if p.cfg.ResetOnExtinction {
				newPop, err := p.reproduction.CreateNew(p.cfg.PopulationSize)
				if err != nil {
					return err
				}
				p.population = newPop
			} else {
				return fmt.Errorf("complete extinction")
			}
		}

		p.species.Speciate(p.cfg, p.population, generation)

		generation++
	}

	return nil
}
