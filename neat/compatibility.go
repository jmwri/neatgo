package neat

import (
	"cmp"
	"math"
	"slices"
)

// gene is one historical marking paired with the parameter it carries: a node's
// bias, or a connection's weight.
//
// An input or bias node has no parameter: it is a fixed signal source that
// every genome in the run shares. It still counts as a gene when matching, but
// its value is left out of the mean parameter difference. Averaging a zero
// difference in for every input would dilute the bias term by however many
// inputs the problem has, so that the same coefficient meant something
// different on every task.
type gene struct {
	id    int
	value float64
	fixed bool
}

// geneIndex holds a genome's genes sorted by historical marking.
//
// Sorting is the whole point of an innovation number. Once both genomes are in
// marking order, comparing them is a single walk down the two lists in step,
// with no lookup table to build and nothing to allocate. Building a map per
// comparison instead costs two map constructions for every genome-to-species
// test, which at a few hundred genomes and a dozen species is thousands of
// throwaway maps per generation.
type geneIndex struct {
	nodes []gene
	conns []gene
}

func newGeneIndex(genome Genome) geneIndex {
	index := geneIndex{
		nodes: make([]gene, 0, genome.NumNodes()),
		conns: make([]gene, 0, len(genome.Connections)),
	}
	for _, layer := range genome.Layers {
		for _, node := range layer {
			index.nodes = append(index.nodes, gene{id: node.ID, value: node.Bias, fixed: !mutableNode(node)})
		}
	}
	for _, connection := range genome.Connections {
		index.conns = append(index.conns, gene{id: connection.ID, value: connection.Weight})
	}
	byID := func(a, b gene) int { return cmp.Compare(a.id, b.id) }
	slices.SortFunc(index.nodes, byID)
	slices.SortFunc(index.conns, byID)
	return index
}

func (g geneIndex) numGenes() int { return len(g.nodes) + len(g.conns) }

// CompatibleWithSpecies reports whether a genome is similar enough to a
// species' representative to join it.
func CompatibleWithSpecies(pop Population, species Species, genome Genome) bool {
	return CompatibilityDistance(pop.Cfg, genome, species.Representative) <= pop.Cfg.SpeciesCompatThreshold
}

// CompatibilityDistance measures how different two genomes are.
//
// Genes are matched by their historical marking. Genes present in only one
// genome (disjoint and excess) are normalised by the size of the larger genome
// so that a big genome is not automatically far from everything; small genomes
// are not normalised at all, following the paper. The remaining terms are the
// mean parameter difference over the genes both genomes share.
func CompatibilityDistance(cfg Config, a, b Genome) float64 {
	return compatibility(cfg, newGeneIndex(a), newGeneIndex(b))
}

func compatibility(cfg Config, a, b geneIndex) float64 {
	biasDiff, matchedBiases, unmatchedNodes := walkGenes(a.nodes, b.nodes)
	weightDiff, matchedWeights, unmatchedConns := walkGenes(a.conns, b.conns)
	unmatched := unmatchedNodes + unmatchedConns

	normaliser := a.numGenes()
	if b.numGenes() > normaliser {
		normaliser = b.numGenes()
	}
	if normaliser < 20 {
		// The paper leaves small genomes unnormalised.
		normaliser = 1
	}

	distance := cfg.SpeciesCompatExcessCoeff * float64(unmatched) / float64(normaliser)
	if matchedWeights > 0 {
		distance += cfg.SpeciesCompatWeightDiffCoeff * weightDiff / float64(matchedWeights)
	}
	if matchedBiases > 0 {
		distance += cfg.SpeciesCompatBiasDiffCoeff * biasDiff / float64(matchedBiases)
	}
	return distance
}

// walkGenes steps down two marking-ordered gene lists together, summing the
// parameter difference of the genes they share and counting the ones only one
// of them holds. Genes with no parameter are matched but not measured.
// Allocates nothing.
func walkGenes(a, b []gene) (diff float64, matched, unmatched int) {
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i].id == b[j].id:
			if !a[i].fixed {
				diff += math.Abs(a[i].value - b[j].value)
				matched++
			}
			i++
			j++
		case a[i].id < b[j].id:
			// Only a holds this marking.
			unmatched++
			i++
		default:
			// Only b holds this marking.
			unmatched++
			j++
		}
	}
	// Whatever is left in either list is excess.
	unmatched += (len(a) - i) + (len(b) - j)
	return diff, matched, unmatched
}
