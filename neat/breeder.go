package neat

import "github.com/jmwri/neatgo/v2/network"

// Breeder produces genomes: it bundles the three things every mutation needs
// together, the settings, the run's random source, and the innovation registry.
//
// These belong in one place because two of them are mutable run state, not
// configuration. Hanging the innovation registry off Config makes a Config
// look reusable when it is not: building a second population from the same
// Config would silently continue the first one's gene-ID sequence, so replaying
// a seeded run in the same process would not reproduce it. Keeping the state
// here, owned by the population, means a Config stays a plain description of
// what to do.
type Breeder struct {
	cfg         Config
	rng         *Rand
	innovations *Innovations
	// scratch is reused by Crossover. It is owned by this Breeder alone: the
	// copies made by withRand and withConfig get their own, which is what
	// keeps the parallel breeding workers from sharing it.
	scratch *crossoverScratch
}

// crossoverScratch is the less fit parent's genes indexed by historical
// marking, kept between children so Crossover does not allocate per child.
type crossoverScratch struct {
	nodes       map[int]network.Node
	connections map[int]network.Connection
}

// NewBreeder returns a Breeder. Passing a nil innovation registry or random
// source creates fresh ones, seeded from cfg.
func NewBreeder(cfg Config, rng *Rand, innovations *Innovations) *Breeder {
	if rng == nil {
		seed := cfg.Seed
		if seed == 0 {
			seed = RandomSeed()
		}
		rng = NewRand(seed)
	}
	if innovations == nil {
		innovations = NewInnovations(NewSequentialIDProvider())
	}
	return &Breeder{cfg: cfg, rng: rng, innovations: innovations}
}

// Config returns the settings this Breeder was built with.
func (b *Breeder) Config() Config { return b.cfg }

// Rand returns the random source. It is not safe for concurrent use.
func (b *Breeder) Rand() *Rand { return b.rng }

// Innovations returns the registry assigning historical markings.
func (b *Breeder) Innovations() *Innovations { return b.innovations }

// withConfig returns a Breeder with updated settings, sharing the random source
// and innovation registry. Used when a generation adjusts a setting such as the
// compatibility threshold.
func (b *Breeder) withConfig(cfg Config) *Breeder {
	return &Breeder{cfg: cfg, rng: b.rng, innovations: b.innovations}
}

// withRand returns a Breeder drawing from a different random source, sharing
// the settings and the innovation registry. Each offspring bred in parallel
// gets its own source so the result does not depend on which worker got there
// first.
func (b *Breeder) withRand(rng *Rand) *Breeder {
	return &Breeder{cfg: b.cfg, rng: rng, innovations: b.innovations}
}
