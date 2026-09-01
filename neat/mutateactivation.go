package neat

import (
	"github.com/jmwri/neatgo/v2/network"
)

// MutateNodeActivations returns a copy of genome with mutated hidden node
// activation functions.
func (b *Breeder) MutateNodeActivations(genome Genome) Genome {
	return b.mutateNodeActivations(CopyGenome(genome))
}

func (b *Breeder) mutateNodeActivations(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	for i, layer := range genome.Layers {
		for j, node := range layer {
			// Only hidden nodes. Input and bias nodes are plain signal
			// sources, and the output activation is fixed by the config
			// because it defines the range the caller reads results in.
			if node.Type != network.Hidden {
				continue
			}
			if !cfg.Chance(rng, cfg.ActivationMutationRate) {
				continue
			}
			genome.Layers[i][j].ActivationFn = network.RandomActivationFunction(rng, cfg.HiddenActivationFns...)
		}
	}
	return genome
}
