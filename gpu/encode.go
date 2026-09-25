package gpu

import (
	"github.com/jmwri/neatgo/v2/network"
)

// nodeRecord and edgeRecord are the layouts the kernel reads: 16 and 8 bytes,
// with no padding.
type nodeRecord struct {
	op        int32
	bias      float32
	firstEdge int32
	endEdge   int32
}

type edgeRecord struct {
	from   int32
	weight float32
}

// netRecord is three int32s: the index of the network's first node, its node
// count, and the index of its first entry in the output table.
const netRecordInts = 3

// encoded is a set of networks laid out end to end for the kernel.
type encoded struct {
	nets     []int32
	nodes    []nodeRecord
	edges    []edgeRecord
	outIdx   []int32
	maxNodes int
}

// canRun reports whether the GPU can compute this network exactly as the CPU
// would, up to precision: every activation must be one the kernel implements,
// and unchanged since registration.
func canRun(program network.Program) bool {
	for _, node := range program.Nodes {
		if node.Kind != network.ProgramComputed {
			continue
		}
		if _, ok := activationOps[node.Activation]; !ok || !network.IsBuiltinActivation(node.Activation) {
			return false
		}
	}
	return true
}

// encode appends one network's program.
func (e *encoded) add(program network.Program) {
	nodeStart, edgeStart := int32(len(e.nodes)), int32(len(e.edges))
	e.nets = append(e.nets, nodeStart, int32(len(program.Nodes)), int32(len(e.outIdx)))
	for _, node := range program.Nodes {
		record := nodeRecord{firstEdge: edgeStart + int32(node.FirstEdge), endEdge: edgeStart + int32(node.EndEdge)}
		switch node.Kind {
		case network.ProgramInput:
			record.op = opInput + int32(node.InputPos)
		case network.ProgramBias:
			record.op = opBias
		default:
			record.op = activationOps[node.Activation]
			record.bias = float32(node.Bias)
		}
		e.nodes = append(e.nodes, record)
	}
	for _, edge := range program.Edges {
		e.edges = append(e.edges, edgeRecord{from: int32(edge.From), weight: float32(edge.Weight)})
	}
	for _, out := range program.Outputs {
		e.outIdx = append(e.outIdx, int32(out))
	}
	e.maxNodes = max(e.maxNodes, len(program.Nodes))
}
