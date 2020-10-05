package neat

import "neatgo"

type Gene interface {
	neatgo.Identifier
	Copy() Gene
}

type NodeGene interface {
	Gene
}

type ConnectionGene interface {
	Gene
}
