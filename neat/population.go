package neat

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
	"sync/atomic"
)

// GeneratePopulation builds the initial population.
//
// Every member is a copy of one template genome with freshly drawn weights and
// biases. That shared genotype matters: NEAT identifies genes by their
// historical marking, so if each genome were generated independently they would
// share no gene IDs at all, every genome would look maximally different from
// every other, and the population would shatter into one species per genome
// before evolution had run a single step.
func GeneratePopulation(cfg Config) (Population, error) {
	if err := cfg.Validate(); err != nil {
		return Population{}, err
	}
	seed := cfg.Seed
	if seed == 0 {
		// Draw one rather than leaving the run un-seeded, and record it below so
		// an interesting result can be replayed by setting Config.Seed.
		seed = RandomSeed()
	}
	pop := Population{
		Cfg:                   cfg,
		Seed:                  seed,
		Breeder:               NewBreeder(cfg, NewRand(seed), nil),
		Genomes:               make([]Genome, cfg.PopulationSize),
		GenomeFitness:         make([]float64, cfg.PopulationSize),
		GenomeAdjustedFitness: make([]float64, cfg.PopulationSize),
		Species:               make([]Species, 0),
		Generation:            0,
		BestEverGenomeFitness: math.Inf(-1),
		BestGenomeFitness:     math.Inf(-1),
	}

	template, err := pop.Breeder.NewGenome()
	if err != nil {
		return pop, fmt.Errorf("neat: failed to generate genome: %w", err)
	}
	for i := range pop.Genomes {
		pop.Genomes[i] = pop.Breeder.RandomizeWeights(template)
	}
	return pop, nil
}

type Population struct {
	Cfg Config
	// Seed is the seed this run was started from. Setting Config.Seed to it
	// reproduces the run exactly.
	Seed uint64
	// Breeder owns the run's random source and innovation registry, and
	// produces every new genome. It must not be used concurrently.
	Breeder *Breeder
	Genomes []Genome
	// GenomeFitness holds the raw fitness reported by the evaluator.
	GenomeFitness []float64
	// GenomeAdjustedFitness holds the fitness after sharing within a species.
	GenomeAdjustedFitness []float64
	Species               []Species
	Generation            int
	BestEverGenome        Genome
	BestEverGenomeFitness float64
	BestGenome            Genome
	BestGenomeFitness     float64
}

// RunGeneration evaluates every genome concurrently and returns the next
// generation.
//
// The evaluator is called once per genome, from one of cfg.Parallelism worker
// goroutines. If any evaluation fails the generation is abandoned: the error is
// returned and the population comes back without having advanced, so the caller
// can fix the problem and retry.
func RunGeneration(ctx context.Context, pop Population, eval Evaluator) (Population, error) {
	if err := Evaluate(ctx, pop, eval); err != nil {
		return pop, err
	}
	return Advance(pop), nil
}

// Advance produces the next generation from the fitnesses already in
// pop.GenomeFitness.
//
// RunGeneration is Evaluate followed by Advance. Call them separately when
// scoring cannot be expressed as one independent Evaluator per genome - a
// competitive tournament where genomes are played off against each other, for
// instance. Write each genome's score into pop.GenomeFitness by index, then
// hand the population here.
func Advance(pop Population) Population {
	pop.Generation++
	// Keep the breeder's view of the settings in step with the population's, so
	// that a caller adjusting pop.Cfg between generations is actually obeyed
	// rather than silently ignored by reproduction, and give the generation its
	// own random stream derived from the run's seed.
	pop.Breeder = pop.Breeder.withConfig(pop.Cfg).withRand(NewRand(generationSeed(pop.Seed, pop.Generation)))
	pop = sanitiseFitness(pop)

	// Record the best of this generation before reproduction replaces the
	// population. Doing this afterwards would report the fitness of whatever
	// happened to be carried into the next generation, most of which has not
	// been evaluated yet.
	pop.BestGenomeFitness = math.Inf(-1)
	for i, fitness := range pop.GenomeFitness {
		if fitness > pop.BestGenomeFitness {
			pop.BestGenomeFitness = fitness
			pop.BestGenome = pop.Genomes[i]
		}
	}
	if pop.BestGenomeFitness > pop.BestEverGenomeFitness {
		pop.BestEverGenomeFitness = pop.BestGenomeFitness
		pop.BestEverGenome = CopyGenome(pop.BestGenome)
	}

	pop = Speciate(pop)
	pop = AdjustCompatThreshold(pop)
	pop = RankSpecies(pop)
	pop = FitnessSharing(pop)
	pop = KillStaleSpecies(pop)
	pop = CullSpecies(pop)
	pop = Evolve(pop)

	return pop
}

// RunOptions controls the loop driven by Run.
type RunOptions struct {
	// MaxGenerations stops the run once pop.Generation reaches this value.
	// Zero means no limit, in which case Solved, OnGeneration or context
	// cancellation must end the run.
	MaxGenerations int
	// Solved is called after each generation. Returning true ends the run.
	Solved func(pop Population) bool
	// OnGeneration is called after each generation, for logging or
	// checkpointing. Returning an error ends the run, and Run returns it.
	OnGeneration func(pop Population) error
}

// Run evolves the population until Solved reports success, MaxGenerations is
// reached, OnGeneration returns an error, or ctx is cancelled.
//
// The population is always returned, including when the run ends early, so the
// best genome found so far is never lost to an error or a cancelled context.
func Run(ctx context.Context, pop Population, eval Evaluator, opts RunOptions) (Population, error) {
	if opts.MaxGenerations < 0 {
		return pop, fmt.Errorf("%w: MaxGenerations must not be negative", ErrNoStopCondition)
	}
	if opts.MaxGenerations == 0 && opts.Solved == nil && opts.OnGeneration == nil && ctx.Done() == nil {
		return pop, fmt.Errorf("%w: set MaxGenerations, Solved, OnGeneration or a cancellable context", ErrNoStopCondition)
	}

	for opts.MaxGenerations == 0 || pop.Generation < opts.MaxGenerations {
		var err error
		pop, err = RunGeneration(ctx, pop, eval)
		if err != nil {
			return pop, err
		}
		if opts.OnGeneration != nil {
			if err := opts.OnGeneration(pop); err != nil {
				return pop, err
			}
		}
		if opts.Solved != nil && opts.Solved(pop) {
			return pop, nil
		}
	}
	return pop, nil
}

// sanitiseFitness replaces any non-finite fitness with a finite one: NaN and
// -Inf become the lowest finite fitness in the population, +Inf the highest.
//
// A non-finite fitness would otherwise propagate: shifting fitnesses to be
// non-negative turns a single infinity into an infinite range, which makes
// the offspring allocation produce garbage for every genome, not just the
// broken one. +Inf is kept at the top rather than sent to the bottom because
// an evaluator that returns it means "cannot be beaten", and turning that
// into the worst score in the population would silently invert the one
// result it was most sure of.
func sanitiseFitness(pop Population) Population {
	lowest, highest := math.Inf(1), math.Inf(-1)
	for _, fitness := range pop.GenomeFitness {
		if math.IsInf(fitness, 0) || math.IsNaN(fitness) {
			continue
		}
		lowest = math.Min(lowest, fitness)
		highest = math.Max(highest, fitness)
	}
	if math.IsInf(lowest, 0) {
		// Nothing finite to fall back on.
		lowest, highest = 0, 0
	}
	for i, fitness := range pop.GenomeFitness {
		switch {
		case math.IsInf(fitness, 1):
			pop.GenomeFitness[i] = highest
		case math.IsInf(fitness, -1) || math.IsNaN(fitness):
			pop.GenomeFitness[i] = lowest
		}
	}
	return pop
}

// worstFitness is the lowest fitness in the population, the floor that
// selection measures fitness from so a negative fitness function still gives
// every genome a non-negative share.
func (p Population) worstFitness() float64 {
	worst := math.Inf(1)
	for _, fitness := range p.GenomeFitness {
		worst = math.Min(worst, fitness)
	}
	return worst
}

// Evolve replaces the population with the next generation, allocating
// offspring to each species in proportion to its adjusted fitness.
//
// The result always contains exactly cfg.PopulationSize genomes.
//
// Breeding runs in parallel; the structural mutations that follow it run one
// genome at a time. That split is what lets the expensive part use every core
// without making the run irreproducible: a structural mutation draws a
// historical marking from the shared innovation registry, and if two workers
// raced for those the gene numbering - and with it speciation, and with it the
// whole run - would depend on which goroutine won.
func Evolve(pop Population) Population {
	offspringCounts := getDesiredOffspringCount(pop)

	newGenomes := make([]Genome, 0, pop.Cfg.PopulationSize)
	newSpecies := make([]Species, 0, len(pop.Species))
	// sourceSpecies[n] is the index in pop.Species that newSpecies[n] came
	// from, so that topping the population up breeds from the right parents.
	sourceSpecies := make([]int, 0, len(pop.Species))
	// Offspring slots to fill, collected first and bred afterwards.
	var slots []offspringSlot

	for i, species := range pop.Species {
		numOffspring := offspringCounts[i]
		if numOffspring <= 0 || len(species.Genomes) == 0 {
			continue
		}

		elitism := pop.Cfg.Elitism
		if elitism > len(species.Genomes) {
			elitism = len(species.Genomes)
		}
		if elitism >= numOffspring && numOffspring > 1 {
			// Never let elitism consume a species' entire allowance. A species
			// made up only of unmutated copies of itself cannot search, and
			// with many small species that stalls the whole population.
			elitism = numOffspring - 1
		}
		if elitism > numOffspring {
			elitism = numOffspring
		}

		speciesGenomes := make([]int, 0, numOffspring)

		// Carry the species' best over untouched, so a solution once found is
		// never lost to a bad mutation.
		for j := 0; j < elitism; j++ {
			speciesGenomes = append(speciesGenomes, len(newGenomes))
			newGenomes = append(newGenomes, pop.Genomes[species.Genomes[j]])
		}

		// Reserve the rest of the allowance for mutated offspring.
		for j := elitism; j < numOffspring; j++ {
			speciesGenomes = append(speciesGenomes, len(newGenomes))
			slots = append(slots, offspringSlot{index: len(newGenomes), species: species})
			newGenomes = append(newGenomes, Genome{})
		}

		species.Genomes = speciesGenomes
		newSpecies = append(newSpecies, species)
		sourceSpecies = append(sourceSpecies, i)
	}

	// Guard the population size against rounding and against every species
	// dying out at once.
	// Species were appended in order, so the last species always owns the
	// tail of newGenomes: dropping the last genome is dropping that species'
	// last member, and a species left with no members goes with it.
	for len(newGenomes) > pop.Cfg.PopulationSize {
		last := len(newSpecies) - 1
		if last < 0 {
			newGenomes = newGenomes[:pop.Cfg.PopulationSize]
			break
		}
		members := newSpecies[last].Genomes
		newSpecies[last].Genomes = members[:len(members)-1]
		newGenomes = newGenomes[:len(newGenomes)-1]
		slots = dropSlot(slots, len(newGenomes))
		if len(newSpecies[last].Genomes) == 0 {
			newSpecies = newSpecies[:last]
			sourceSpecies = sourceSpecies[:last]
		}
	}
	for len(newGenomes) < pop.Cfg.PopulationSize {
		if len(newSpecies) == 0 {
			// Total extinction: reseed from the best genome ever seen, falling
			// back to any surviving genome if there is no best yet.
			seed := pop.BestEverGenome
			if seed.NumLayers() == 0 {
				if len(pop.Genomes) == 0 {
					break
				}
				seed = pop.Genomes[0]
			}
			newGenomes = append(newGenomes, pop.Breeder.MutateGenome(seed))
			continue
		}
		// Top up from the fittest surviving species.
		newSpecies[0].Genomes = append(newSpecies[0].Genomes, len(newGenomes))
		slots = append(slots, offspringSlot{index: len(newGenomes), species: pop.Species[sourceSpecies[0]]})
		newGenomes = append(newGenomes, Genome{})
	}

	fillOffspring(pop, slots, newGenomes)

	pop.Genomes = newGenomes
	pop.GenomeFitness = make([]float64, len(newGenomes))
	pop.GenomeAdjustedFitness = make([]float64, len(newGenomes))
	pop.Species = newSpecies
	return pop
}

// offspringSlot is a place in the next generation waiting for a child.
//
// Each slot carries its own pair of random seeds rather than a random source,
// so that breeding a slot allocates nothing: a worker reseeds the source it
// already owns. Two seeds because breeding and structural mutation happen in
// separate passes, and each needs a stream that depends only on the run's seed.
type offspringSlot struct {
	index      int
	species    Species
	breedSeed  uint64
	mutateSeed uint64
	genome     Genome
}

func dropSlot(slots []offspringSlot, index int) []offspringSlot {
	for i := range slots {
		if slots[i].index == index {
			return append(slots[:i], slots[i+1:]...)
		}
	}
	return slots
}

// fillOffspring breeds every reserved slot and writes the results into genomes.
func fillOffspring(pop Population, slots []offspringSlot, genomes []Genome) {
	if len(slots) == 0 {
		return
	}

	// Draw every slot's seeds up front, in order, from the population's own
	// source. They therefore depend on the run's seed and nothing else - not on
	// how the work is later divided between goroutines.
	for i := range slots {
		slots[i].breedSeed = pop.Breeder.rng.Uint64()
		slots[i].mutateSeed = pop.Breeder.rng.Uint64()
	}

	// Breed in parallel. This reads the previous generation, which nothing is
	// writing to, and otherwise touches only the worker's own state.
	floor := pop.worstFitness()
	workers := reproductionWorkers(pop.Cfg.Parallelism, len(slots))
	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			source := rand.NewPCG(0, 0)
			breeder := pop.Breeder.withRand(rand.New(source))
			for {
				i := int(next.Add(1)) - 1
				if i >= len(slots) {
					return
				}
				source.Seed(slots[i].breedSeed, slots[i].breedSeed^pcgStreamOffset)
				slots[i].genome = breeder.breed(pop, slots[i].species, floor)
			}
		}()
	}
	wg.Wait()

	// Apply structural mutations one at a time, in slot order, so historical
	// markings are handed out in the same order on every run.
	source := rand.NewPCG(0, 0)
	breeder := pop.Breeder.withRand(rand.New(source))
	for i := range slots {
		source.Seed(slots[i].mutateSeed, slots[i].mutateSeed^pcgStreamOffset)
		genomes[slots[i].index] = breeder.mutateStructure(slots[i].genome)
	}
}

// reproductionWorkers sizes the breeding pool. Breeding is pure computation, so
// Unlimited is treated as one worker per core rather than one per genome:
// unlike a fitness evaluator, it never blocks on anything.
func reproductionWorkers(parallelism, slots int) int {
	if parallelism == Unlimited || parallelism < 0 {
		parallelism = 0
	}
	return Workers(parallelism, slots)
}
