package neat_test

import (
	"errors"
	"github.com/jmwri/neatgo/activation"
	"github.com/jmwri/neatgo/aggregation"
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/net"
	"math/rand"
	"testing"
)

func TestNewNodeGene(t *testing.T) {
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	if gene.ID() != 10 {
		t.Errorf("expected %d, got %d", 10, gene.ID())
	}
	if gene.Node() != node {
		t.Errorf("didnt get expected node")
	}
}

func TestNewNodeGeneFromCrossover_FailWhenNotSameID(t *testing.T) {
	nodeA := net.NewNode(10, .2, activation.Nil, aggregation.Sum)
	geneA := neat.NewNodeGene(nodeA)
	nodeB := net.NewNode(12, .7, activation.Nil, aggregation.Sum)
	geneB := neat.NewNodeGene(nodeB)
	_, err := neat.NewNodeGeneFromCrossover(geneA, geneB)
	if err == nil {
		t.Fatalf("expected error: %s", err)
	}
	if !errors.Is(err, neat.ErrCrossoverNotSameParent) {
		t.Fatalf("expected %s, got %s", neat.ErrCrossoverNotSameParent, err)
	}
}

func TestNewNodeGeneFromCrossover(t *testing.T) {
	rand.Seed(1234)
	nodeA := net.NewNode(10, .2, activation.Nil, aggregation.Sum)
	geneA := neat.NewNodeGene(nodeA)
	nodeB := net.NewNode(10, .7, activation.Nil, aggregation.Sum)
	geneB := neat.NewNodeGene(nodeB)
	// The rand seed should take all props from geneB
	child, err := neat.NewNodeGeneFromCrossover(geneA, geneB)
	if err != nil {
		t.Fatalf("failed to crossover node gene: %s", err)
	}

	if child.ID() != 10 {
		t.Errorf("expected %d, got %d", 10, child.ID())
	}

	if child.Node().Bias() != .7 {
		t.Errorf("expected %f, got %f", .7, child.Node().Bias())
	}

	if !activation.IsSameFunction(child.Node().ActivationFn(), activation.Nil) {
		t.Errorf("got wrong activation fn")
	}

	if !aggregation.IsSameFunction(child.Node().AggregationFn(), aggregation.Sum) {
		t.Errorf("got wrong aggregation fn")
	}
}

func getNodeGeneMutateConfig() *neat.Config {
	return &neat.Config{
		BiasMutateRate:        0,
		BiasReplaceRate:       0,
		BiasMinValue:          0,
		BiasMaxValue:          1,
		BiasMutatePower:       .2,
		ActivationMutateRate:  0,
		ActivationOptions:     activation.FnAll,
		AggregationMutateRate: 0,
		AggregationOptions:    aggregation.FnAll,
	}
}

func TestNodeGene_Mutate_Bias_NoMutation(t *testing.T) {
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	cfg := getNodeGeneMutateConfig()
	// Set mutate rate to 0, should never mutate
	cfg.BiasMutateRate = 0

	gene.Mutate(cfg)
	if gene.Node().Bias() != .5 {
		t.Errorf("expected %v, got %v", .5, gene.Node().Bias())
	}
}

func TestNodeGene_Mutate_Bias_Replace(t *testing.T) {
	rand.Seed(1234)
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	cfg := getNodeGeneMutateConfig()
	// Set mutate rate to 1, should always mutate
	cfg.BiasMutateRate = 1
	// Should always replace
	cfg.BiasReplaceRate = 1

	gene.Mutate(cfg)
	// Rand seed means the randomly generated bias is always this value
	expectedBias := 0.6511888420060171
	if gene.Node().Bias() != expectedBias {
		t.Errorf("expected %v, got %v", expectedBias, gene.Node().Bias())
	}
}

func TestNodeGene_Mutate_Bias_Power(t *testing.T) {
	rand.Seed(1234)
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	cfg := getNodeGeneMutateConfig()
	// Set mutate rate to 1, should always mutate
	cfg.BiasMutateRate = 1
	// Should never replace
	cfg.BiasReplaceRate = 0

	gene.Mutate(cfg)
	// Rand seed means the randomly generated bias is always this value
	expectedBias := 0.5604755368024068
	if gene.Node().Bias() != expectedBias {
		t.Errorf("expected %v, got %v", expectedBias, gene.Node().Bias())
	}
}

func TestNodeGene_Mutate_ActivationFn_NoMutation(t *testing.T) {
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	cfg := getNodeGeneMutateConfig()
	// Set mutate rate to 0, should never mutate
	cfg.ActivationMutateRate = 0
	oldActivationFn := gene.Node().ActivationFn()
	gene.Mutate(cfg)
	newActivationFn := gene.Node().ActivationFn()

	if !activation.IsSameFunction(oldActivationFn, newActivationFn) {
		t.Errorf("activationFn was mutated")
	}
}

func TestNodeGene_Mutate_ActivationFn(t *testing.T) {
	rand.Seed(1234)
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	cfg := getNodeGeneMutateConfig()
	// Set mutate rate to 1, should always mutate
	cfg.ActivationMutateRate = 1
	gene.Mutate(cfg)
	newActivationFn := gene.Node().ActivationFn()

	if !activation.IsSameFunction(activation.Cube, newActivationFn) {
		t.Errorf("activationFn was not mutated")
	}
}

func TestNodeGene_Mutate_AggregationFn_NoMutation(t *testing.T) {
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	cfg := getNodeGeneMutateConfig()
	// Set mutate rate to 0, should never mutate
	cfg.AggregationMutateRate = 0
	oldAggregationFn := gene.Node().AggregationFn()
	gene.Mutate(cfg)
	newAggregationFn := gene.Node().AggregationFn()

	if !aggregation.IsSameFunction(oldAggregationFn, newAggregationFn) {
		t.Errorf("aggregationFn was mutated")
	}
}

func TestNodeGene_Mutate_AggregationFn(t *testing.T) {
	rand.Seed(1234)
	node := net.NewNode(10, .5, activation.Nil, aggregation.Sum)
	gene := neat.NewNodeGene(node)

	cfg := getNodeGeneMutateConfig()
	// Set mutate rate to 1, should always mutate
	cfg.AggregationMutateRate = 1
	gene.Mutate(cfg)
	newAggregationFn := gene.Node().AggregationFn()

	if !aggregation.IsSameFunction(aggregation.Mean, newAggregationFn) {
		t.Errorf("activationFn was not mutated")
	}
}

func TestNodeGene_Distance(t *testing.T) {
	nodeA := net.NewNode(10, .2, activation.Nil, aggregation.Sum)
	geneA := neat.NewNodeGene(nodeA)
	nodeB := net.NewNode(10, .8, activation.Nil, aggregation.Sum)
	geneB := neat.NewNodeGene(nodeB)
	cfg := getNodeGeneMutateConfig()
	cfg.CompatibilityWeightCoefficient = 1

	dist := geneA.Distance(geneB, cfg)
	expectedDist := 0.6000000000000001
	if dist != expectedDist {
		t.Errorf("expected %v, got %v", expectedDist, dist)
	}

	dist = geneB.Distance(geneA, cfg)
	if dist != expectedDist {
		t.Errorf("expected %v, got %v", expectedDist, dist)
	}
}
