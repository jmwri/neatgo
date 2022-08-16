package neat

import (
	"github.com/jmwri/neatgo/network"
)

func MutateNodeActivations(cfg Config, genome Genome) Genome {
	genome = CopyGenome(genome)
	for j, layer := range genome.Layers {
		for i, _ := range layer {
			seed := cfg.RandFloatProvider(0, 1)
			if seed > cfg.ActivationMutationRate {
				continue
			}

			newActivation := network.RandomActivationFunction(cfg.HiddenActivationFns...)
			genome.Layers[j][i].ActivationFn = newActivation
		}
	}
	return genome
}
