package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/net"
	"github.com/jmwri/neatgo/util"
	"math"
)

func NewConnectionGene(c *net.Connection) *ConnectionGene {
	return &ConnectionGene{
		c: c,
	}
}

func NewConnectionGeneFromCrossover(parentA *ConnectionGene, parentB *ConnectionGene) (*ConnectionGene, error) {
	if parentA.ID() != parentB.ID() {
		return nil, ErrCrossoverNotSameParent
	}

	newWeight := parentA.c.Weight()
	if util.RandFloat(0, 1) < 0.5 {
		newWeight = parentB.c.Weight()
	}
	newEnabled := parentA.c.Enabled()
	if util.RandFloat(0, 1) < 0.5 {
		newEnabled = parentB.c.Enabled()
	}

	newConn := net.NewConnection(parentA.ID(), newWeight, parentA.c.From(), parentA.c.To(), newEnabled)
	return NewConnectionGene(newConn), nil
}

type ConnectionGene struct {
	c *net.Connection
}

func (g *ConnectionGene) ID() int64 {
	return g.c.ID()
}

func (g *ConnectionGene) Connection() *net.Connection {
	return g.c
}

func (g *ConnectionGene) Copy() *ConnectionGene {
	connCp := net.NewConnection(g.c.ID(), g.c.Weight(), g.c.From(), g.c.To(), g.c.Enabled())
	return NewConnectionGene(connCp)
}

func (g *ConnectionGene) Mutate(cfg *Config) {
	g.mutateWeight(cfg)
	g.mutateEnabled(cfg)
}

func (g *ConnectionGene) mutateWeight(cfg *Config) {
	if util.RandFloat(0, 1) > cfg.WeightMutateRate {
		return
	}
	if util.RandFloat(0, 1) <= cfg.WeightReplaceRate {
		// We should replace the weight entirely
		g.c.SetWeight(util.RandFloat(cfg.WeightMinValue, cfg.WeightMaxValue))
	} else {
		// We should adjust the weight by power
		lowBound := g.c.Weight() - cfg.WeightMutatePower
		if lowBound < cfg.WeightMinValue {
			lowBound = cfg.WeightMinValue
		}
		highBound := g.c.Weight() + cfg.WeightMutatePower
		if highBound > cfg.WeightMaxValue {
			highBound = cfg.WeightMaxValue
		}
		g.c.SetWeight(util.RandFloat(lowBound, highBound))
	}
}

func (g *ConnectionGene) mutateEnabled(cfg *Config) {
	rate := cfg.EnabledMutateRate
	if g.c.Enabled() {
		rate += cfg.EnabledRateToFalseAdd
	} else {
		rate += cfg.EnabledRateToTrueAdd
	}
	if util.RandFloat(0, 1) > rate {
		return
	}
	choiceV := util.RandFloat(0, 1)
	fmt.Println(choiceV)
	g.c.SetEnabled(choiceV < .5)
}

func (g *ConnectionGene) Distance(other *ConnectionGene, cfg *Config) float64 {
	d := math.Abs(g.c.Weight() - other.c.Weight())
	if g.c.Enabled() != other.c.Enabled() {
		d += 1
	}

	return d * cfg.CompatibilityWeightCoefficient
}
