package neat_test

import (
	"errors"
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/net"
	"math/rand"
	"testing"
)

func TestNewConnectionGene(t *testing.T) {
	conn := net.NewConnection(10, .5, 2, 4, true)
	gene := neat.NewConnectionGene(conn)

	if gene.ID() != 10 {
		t.Errorf("expected %d, got %d", 10, gene.ID())
	}
	if gene.Connection() != conn {
		t.Errorf("got wrong connection out of gene")
	}
}

func TestNewConnectionGeneFromCrossover_FailWhenNotSameID(t *testing.T) {
	connA := net.NewConnection(10, .2, 2, 4, true)
	geneA := neat.NewConnectionGene(connA)
	connB := net.NewConnection(12, .7, 2, 4, true)
	geneB := neat.NewConnectionGene(connB)
	_, err := neat.NewConnectionGeneFromCrossover(geneA, geneB)
	if err == nil {
		t.Fatalf("expected error: %s", err)
	}
	if !errors.Is(err, neat.ErrCrossoverNotSameParent) {
		t.Fatalf("expected %s, got %s", neat.ErrCrossoverNotSameParent, err)
	}
}

func TestNewConnectionGeneFromCrossover(t *testing.T) {
	rand.Seed(1234)
	connA := net.NewConnection(10, .2, 2, 4, true)
	geneA := neat.NewConnectionGene(connA)
	connB := net.NewConnection(10, .7, 2, 4, true)
	geneB := neat.NewConnectionGene(connB)
	// The rand seed should take all props from geneB
	child, err := neat.NewConnectionGeneFromCrossover(geneA, geneB)
	if err != nil {
		t.Fatalf("failed to crossover connection gene: %s", err)
	}

	if child.ID() != 10 {
		t.Errorf("expected %d, got %d", 10, child.ID())
	}

	if !child.Connection().Enabled() {
		t.Errorf("expected %v, got %v", true, child.Connection().Enabled())
	}
	if child.Connection().Weight() != .7 {
		t.Errorf("expected %f, got %f", .7, child.Connection().Weight())
	}
	if child.Connection().From() != 2 {
		t.Errorf("expected %d, got %d", 2, child.Connection().From())
	}
	if child.Connection().To() != 4 {
		t.Errorf("expected %d, got %d", 4, child.Connection().To())
	}
}

func getConnGeneMutateConfig() *neat.Config {
	return &neat.Config{
		WeightMaxValue:        1,
		WeightMinValue:        0,
		WeightMutatePower:     .3,
		WeightMutateRate:      .5,
		WeightReplaceRate:     .2,
		EnabledMutateRate:     .5,
		EnabledRateToFalseAdd: .2,
		EnabledRateToTrueAdd:  .2,
	}
}

func TestConnectionGene_Mutate_Weight_NoMutation(t *testing.T) {
	conn := net.NewConnection(10, .5, 2, 4, true)
	gene := neat.NewConnectionGene(conn)

	cfg := getConnGeneMutateConfig()
	// Set mutate rate to 0, should never mutate
	cfg.WeightMutateRate = 0
	gene.Mutate(cfg)
}

func TestConnectionGene_Mutate_Weight_Replace(t *testing.T) {
	rand.Seed(1234)
	conn := net.NewConnection(10, .5, 2, 4, true)
	gene := neat.NewConnectionGene(conn)

	cfg := getConnGeneMutateConfig()
	// Set mutate rate to 1, should always mutate
	cfg.WeightMutateRate = 1
	// Set mutate replace rate to 1, should always replace
	cfg.WeightReplaceRate = 1
	gene.Mutate(cfg)

	// Rand seed means the randomly generated weight is always this value
	expectedWeight := 0.6511888420060171
	if gene.Connection().Weight() != expectedWeight {
		t.Errorf("expected %v, got %v", expectedWeight, gene.Connection().Weight())
	}
}

func TestConnectionGene_Mutate_Weight_Power(t *testing.T) {
	rand.Seed(1234)
	conn := net.NewConnection(10, .5, 2, 4, true)
	gene := neat.NewConnectionGene(conn)

	cfg := getConnGeneMutateConfig()
	// Set mutate rate to 1, should always mutate
	cfg.WeightMutateRate = 1
	// Set mutate replace rate to 1, should always replace
	cfg.WeightReplaceRate = 0
	gene.Mutate(cfg)

	// Rand seed means the randomly generated weight is always this value
	expectedWeight := 0.5907133052036103
	if gene.Connection().Weight() != expectedWeight {
		t.Errorf("expected %v, got %v", expectedWeight, gene.Connection().Weight())
	}
}

func TestConnectionGene_Mutate_Enabled_NoMutation(t *testing.T) {
	conn := net.NewConnection(10, .5, 2, 4, true)
	gene := neat.NewConnectionGene(conn)

	cfg := getConnGeneMutateConfig()
	// Set mutate rate to 0, should never mutate
	cfg.EnabledMutateRate = 0
	gene.Mutate(cfg)

	if gene.Connection().Enabled() != true {
		t.Errorf("expected %v, got %v", true, gene.Connection().Enabled())
	}
}

func TestConnectionGene_Mutate_Enabled_ToFalse(t *testing.T) {
	rand.Seed(1)
	conn := net.NewConnection(10, .5, 2, 4, true)
	gene := neat.NewConnectionGene(conn)

	cfg := getConnGeneMutateConfig()
	// Set mutate rate to 0, should always mutate
	cfg.EnabledMutateRate = 1
	cfg.EnabledRateToFalseAdd = 0

	// The rand Seed makes sure that the second random number is >= .5
	gene.Mutate(cfg)

	if gene.Connection().Enabled() != false {
		t.Errorf("expected %v, got %v", false, gene.Connection().Enabled())
	}
}

func TestConnectionGene_Mutate_Enabled_ToTrue(t *testing.T) {
	rand.Seed(4)
	conn := net.NewConnection(10, .5, 2, 4, false)
	gene := neat.NewConnectionGene(conn)

	cfg := getConnGeneMutateConfig()
	// Set mutate rate to 1, should always mutate
	cfg.EnabledMutateRate = 1
	cfg.EnabledRateToTrueAdd = 0

	// The rand Seed makes sure that the second random number is < .5
	gene.Mutate(cfg)

	if gene.Connection().Enabled() != true {
		t.Errorf("expected %v, got %v", true, gene.Connection().Enabled())
	}
}

func TestConnectionGene_Distance(t *testing.T) {
	connA := net.NewConnection(10, .2, 2, 4, true)
	geneA := neat.NewConnectionGene(connA)
	connB := net.NewConnection(10, .7, 2, 4, false)
	geneB := neat.NewConnectionGene(connB)
	cfg := getConnGeneMutateConfig()
	cfg.CompatibilityWeightCoefficient = 1

	dist := geneA.Distance(geneB, cfg)
	expectedDist := 1.5
	if dist != expectedDist {
		t.Errorf("expected %v, got %v", expectedDist, dist)
	}

	dist = geneB.Distance(geneA, cfg)
	if dist != expectedDist {
		t.Errorf("expected %v, got %v", expectedDist, dist)
	}
}
