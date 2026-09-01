package neat

// MutateNodeBiases returns a copy of genome with perturbed node biases.
func (b *Breeder) MutateNodeBiases(genome Genome) Genome {
	return b.mutateNodeBiases(CopyGenome(genome))
}

func (b *Breeder) mutateNodeBiases(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	for i, layer := range genome.Layers {
		for j, node := range layer {
			// Input and bias nodes have no learnable bias.
			if !mutableNode(node) {
				continue
			}
			if !cfg.Chance(rng, cfg.BiasMutationRate) {
				continue
			}
			if cfg.Chance(rng, cfg.BiasReplaceRate) {
				genome.Layers[i][j].Bias = cfg.RandBias(rng)
				continue
			}
			genome.Layers[i][j].Bias = cfg.PerturbBias(rng, node.Bias)
		}
	}
	return genome
}
