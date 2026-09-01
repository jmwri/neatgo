package neat

import (
	"fmt"
	"math"
)

// Snapshot is a complete, serialisable record of a run in progress.
//
// Every field is exported and made of plain types, so a Snapshot round-trips
// through encoding/json without any custom marshalling. Saving one after each
// generation is enough to survive a crash on a long run.
//
// There is deliberately no random state here. A generation's randomness is
// derived from Seed and Generation, so a restored run continues exactly as the
// uninterrupted one would have.
type Snapshot struct {
	Cfg                   Config
	Seed                  uint64
	Generation            int
	Genomes               []Genome
	Species               []Species
	GenomeFitness         []float64
	BestGenome            Genome
	BestGenomeFitness     float64
	BestEverGenome        Genome
	BestEverGenomeFitness float64
	Innovations           InnovationsSnapshot
}

// InnovationsSnapshot records which historical markings have been handed out.
//
// This has to travel with the population. Restoring the genomes but not the
// registry would let a later mutation issue a marking that is already in use,
// and two different structures sharing a marking is precisely what historical
// markings exist to prevent.
type InnovationsSnapshot struct {
	CurrentID   int
	Connections []ConnectionInnovation
	Splits      []SplitInnovation
}

// ConnectionInnovation is the marking issued for a connection between two nodes.
type ConnectionInnovation struct {
	From, To int
	ID       int
}

// SplitInnovation is the set of markings issued for splitting a connection.
type SplitInnovation struct {
	ConnectionID int
	Split        Split
}

// Snapshot captures the run so it can be saved and later restored.
func (p Population) Snapshot() Snapshot {
	return Snapshot{
		Cfg:                   p.Cfg,
		Seed:                  p.Seed,
		Generation:            p.Generation,
		Genomes:               p.Genomes,
		Species:               p.Species,
		GenomeFitness:         p.GenomeFitness,
		BestGenome:            p.BestGenome,
		BestGenomeFitness:     p.BestGenomeFitness,
		BestEverGenome:        p.BestEverGenome,
		BestEverGenomeFitness: p.BestEverGenomeFitness,
		Innovations:           p.Breeder.Innovations().Snapshot(),
	}
}

// Restore rebuilds a population from a snapshot, ready to carry on from the
// generation it was taken at.
func Restore(snapshot Snapshot) (Population, error) {
	if err := snapshot.Cfg.Validate(); err != nil {
		return Population{}, err
	}
	if len(snapshot.Genomes) == 0 {
		return Population{}, fmt.Errorf("%w: snapshot has no genomes", ErrInvalidConfig)
	}
	for i, species := range snapshot.Species {
		for _, genomeIndex := range species.Genomes {
			if genomeIndex < 0 || genomeIndex >= len(snapshot.Genomes) {
				return Population{}, fmt.Errorf("%w: species %d refers to genome %d, which does not exist",
					ErrInvalidConfig, i, genomeIndex)
			}
		}
	}

	seed := snapshot.Seed
	if seed == 0 {
		seed = RandomSeed()
	}

	pop := Population{
		Cfg:                   snapshot.Cfg,
		Seed:                  seed,
		Breeder:               NewBreeder(snapshot.Cfg, NewRand(generationSeed(seed, snapshot.Generation)), RestoreInnovations(snapshot.Innovations)),
		Genomes:               snapshot.Genomes,
		GenomeFitness:         snapshot.GenomeFitness,
		GenomeAdjustedFitness: make([]float64, len(snapshot.Genomes)),
		Species:               snapshot.Species,
		Generation:            snapshot.Generation,
		BestGenome:            snapshot.BestGenome,
		BestGenomeFitness:     snapshot.BestGenomeFitness,
		BestEverGenome:        snapshot.BestEverGenome,
		BestEverGenomeFitness: snapshot.BestEverGenomeFitness,
	}
	if len(pop.GenomeFitness) != len(pop.Genomes) {
		pop.GenomeFitness = make([]float64, len(pop.Genomes))
	}
	if pop.BestGenomeFitness == 0 && pop.BestGenome.NumLayers() == 0 {
		pop.BestGenomeFitness = math.Inf(-1)
	}
	if pop.BestEverGenomeFitness == 0 && pop.BestEverGenome.NumLayers() == 0 {
		pop.BestEverGenomeFitness = math.Inf(-1)
	}
	return pop, nil
}

// Snapshot records every marking issued so far.
func (i *Innovations) Snapshot() InnovationsSnapshot {
	i.mu.Lock()
	defer i.mu.Unlock()

	snapshot := InnovationsSnapshot{
		CurrentID:   i.ids.Current(),
		Connections: make([]ConnectionInnovation, 0, len(i.connections)),
		Splits:      make([]SplitInnovation, 0, len(i.splits)),
	}
	for key, id := range i.connections {
		snapshot.Connections = append(snapshot.Connections, ConnectionInnovation{From: key.from, To: key.to, ID: id})
	}
	for connectionID, split := range i.splits {
		snapshot.Splits = append(snapshot.Splits, SplitInnovation{ConnectionID: connectionID, Split: split})
	}
	return snapshot
}

// RestoreInnovations rebuilds a registry from a snapshot.
func RestoreInnovations(snapshot InnovationsSnapshot) *Innovations {
	ids := NewSequentialIDProvider()
	ids.SetCurrent(snapshot.CurrentID)
	innovations := NewInnovations(ids)
	for _, connection := range snapshot.Connections {
		innovations.connections[connectionKey{from: connection.From, to: connection.To}] = connection.ID
	}
	for _, split := range snapshot.Splits {
		innovations.splits[split.ConnectionID] = split.Split
	}
	return innovations
}
