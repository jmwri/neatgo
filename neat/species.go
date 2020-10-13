package neat

import (
	"log"
	"neatgo/aggregation"
	"neatgo/util"
	"sort"
)

func NewSpecies(id int64, generation int) *Species {
	return &Species{
		id:              id,
		createdGen:      generation,
		lastImprovedGen: 0,
		fitness:         0,
		representative:  nil,
		members:         make([]*Genome, 0),
		adjustedFitness: 0,
		fitnessHistory:  make([]float64, 0),
	}
}

type Species struct {
	id              int64
	createdGen      int
	lastImprovedGen int
	fitness         float64
	representative  *Genome
	members         []*Genome
	adjustedFitness float64
	fitnessHistory  []float64
}

func (s *Species) Update(representative *Genome, members []*Genome) {
	s.representative = representative
	s.members = members
}

func (s *Species) Fitnesses() []float64 {
	fitnesses := make([]float64, len(s.members))
	for i, m := range s.members {
		fitnesses[i] = m.Fitness()
	}
	return fitnesses
}

type GenomeDistanceCacheKey struct {
	A int64
	B int64
}

type SpeciesCandidateGenome struct {
	Distance float64
	G        *Genome
}

type SpeciesCandidateSpeciesID struct {
	Distance  float64
	SpeciesID int64
}

func NewGenomeDistanceCache(cfg *Config) *GenomeDistanceCache {
	return &GenomeDistanceCache{
		cfg:       cfg,
		distances: make(map[GenomeDistanceCacheKey]float64),
		hits:      0,
		misses:    0,
	}
}

type GenomeDistanceCache struct {
	cfg       *Config
	distances map[GenomeDistanceCacheKey]float64
	hits      int
	misses    int
}

func (c *GenomeDistanceCache) Distance(a *Genome, b *Genome) float64 {
	key := GenomeDistanceCacheKey{
		A: a.ID(),
		B: b.ID(),
	}
	if d, ok := c.distances[key]; ok {
		c.hits++
		return d
	}
	c.misses++
	d := a.Distance(b, c.cfg)
	keyInverse := GenomeDistanceCacheKey{
		A: b.ID(),
		B: a.ID(),
	}
	c.distances[keyInverse] = d
	return d
}

func (c *GenomeDistanceCache) Distances() []float64 {
	distances := make([]float64, 0)
	for _, d := range c.distances {
		distances = append(distances, d)
	}
	return distances
}

func NewSpeciesSet(cfg *Config) *SpeciesSet {
	return &SpeciesSet{
		cfg:                 cfg,
		species:             make(map[int64]*Species),
		genomeIDToSpeciesID: make(map[int64]int64),
		currentIndex:        1,
	}
}

type SpeciesSet struct {
	cfg                 *Config
	species             map[int64]*Species
	genomeIDToSpeciesID map[int64]int64
	currentIndex        int64
}

func (ss *SpeciesSet) GetNextIndex() int64 {
	i := ss.currentIndex
	ss.currentIndex++
	return i
}

func (ss *SpeciesSet) GetSpeciesID(genomeID int64) int64 {
	if speciesID, ok := ss.genomeIDToSpeciesID[genomeID]; ok {
		return speciesID
	}
	return -1
}

func (ss *SpeciesSet) GetSpecies(genomeID int64) *Species {
	speciesID := ss.GetSpeciesID(genomeID)
	if speciesID == -1 {
		return nil
	}
	if species, ok := ss.species[speciesID]; ok {
		return species
	}
	return nil
}

func (ss *SpeciesSet) Speciate(cfg *Config, population map[int64]*Genome, generation int) {
	compatThreshold := cfg.CompatibilityThreshold

	unspeciated := make([]int64, 0)
	for speciesID := range ss.species {
		unspeciated = append(unspeciated, speciesID)
	}

	distances := NewGenomeDistanceCache(cfg)
	newRepresentatives := make(map[int64]int64)
	newMembers := make(map[int64][]int64)

	for speciesID, species := range ss.species {
		candidates := make([]SpeciesCandidateGenome, 0)
		for _, genomeID := range unspeciated {
			genome := population[genomeID]
			d := distances.Distance(species.representative, genome)
			candidates = append(candidates, SpeciesCandidateGenome{
				Distance: d,
				G:        genome,
			})
		}

		// Sort candidates by min dist
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})

		newRepresentativeID := candidates[0].G.ID()
		newRepresentatives[speciesID] = newRepresentativeID
		newMembers[speciesID] = []int64{newRepresentativeID}

		unspeciated = util.RemoveInt64FromSlice(unspeciated, newRepresentativeID)
	}

	unspeciatedRemaining := len(unspeciated)
	for unspeciatedRemaining > 0 {
		genomeID := unspeciated[0]
		genome := population[genomeID]
		unspeciated = unspeciated[1:]
		unspeciatedRemaining--

		candidates := make([]SpeciesCandidateSpeciesID, 0)
		for speciesID, representativeID := range newRepresentatives {
			representative := population[representativeID]
			d := distances.Distance(representative, genome)
			if d < compatThreshold {
				candidates = append(candidates, SpeciesCandidateSpeciesID{
					Distance:  d,
					SpeciesID: speciesID,
				})
			}
		}

		if len(candidates) > 0 {
			// Sort candidates by min dist
			sort.Slice(candidates, func(i, j int) bool {
				return candidates[i].Distance < candidates[j].Distance
			})
			speciesID := candidates[0].SpeciesID
			newMembers[speciesID] = append(newMembers[speciesID], genomeID)
		} else {
			// No species is similar enough, create a new species using this genome as it's representative.
			speciesID := ss.GetNextIndex()
			newRepresentatives[speciesID] = genomeID
			newMembers[speciesID] = []int64{genomeID}
		}
	}

	ss.genomeIDToSpeciesID = make(map[int64]int64)
	for speciesID, representativeID := range newRepresentatives {
		species, ok := ss.species[speciesID]
		if !ok {
			species = NewSpecies(speciesID, generation)
			ss.species[speciesID] = species
		}

		members := newMembers[speciesID]
		memberGenomes := make([]*Genome, 0)
		for _, genomeID := range members {
			ss.genomeIDToSpeciesID[genomeID] = speciesID
			memberGenomes = append(memberGenomes, population[genomeID])
		}
		species.Update(population[representativeID], memberGenomes)
	}

	distancesSlice := distances.Distances()
	geneticDistancesMean := aggregation.Mean(distancesSlice)
	geneticDistancesStdev := aggregation.Stdev(distancesSlice)

	log.Printf("genetic distance mean: %f, stdev: %f\n", geneticDistancesMean, geneticDistancesStdev)
}
