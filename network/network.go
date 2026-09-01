package network

import (
	"fmt"
	"sync"
)

// Network is a compiled, ready to run neural network.
//
// Compiling resolves the topological evaluation order, each node's activation
// function and the source of every incoming connection exactly once. Activate
// is then a flat loop over slices: no map lookups, no graph walking, and no
// allocation in the steady state.
//
// A Network is immutable once built and safe for concurrent use by any number
// of goroutines.
type Network struct {
	nodes          []compiledNode // in topological order
	outputs        []int          // index into nodes for each output position
	numInputs      int
	numConnections int
	scratch        sync.Pool
}

type compiledNode struct {
	kind       NodeType
	bias       float64
	activation ActivationFunction
	inputs     []weightedInput
	// inputPos is the position in the input slice, for Input nodes only.
	inputPos int
}

type weightedInput struct {
	// from indexes the value buffer. Because nodes are stored in topological
	// order it is always smaller than the index of the node it feeds, so a
	// single forward pass is enough.
	from   int
	weight float64
}

// Compile builds a runnable network from a genome's nodes and connections.
//
// Connections that are disabled, or that reference a node not in nodes, are
// dropped. Compile fails if the enabled connections contain a cycle, if a node
// uses an unregistered activation function, or if two nodes share an ID.
//
// The graph is held in flat arrays with per-node offsets rather than a slice
// of slices. A slice per node would allocate once per node and again as each
// grew; this allocates a fixed handful of times no matter how large the
// network is, which matters because a population compiles every genome afresh
// each generation.
func Compile(nodes []Node, connections []Connection) (*Network, error) {
	nodeIndex := make(map[int]int, len(nodes))
	for i, node := range nodes {
		if _, duplicate := nodeIndex[node.ID]; duplicate {
			return nil, fmt.Errorf("%w: %d", ErrDuplicateNode, node.ID)
		}
		nodeIndex[node.ID] = i
	}

	// Input and output positions follow the order the nodes were given in, not
	// the evaluation order, so that callers can rely on output[k] being the
	// k-th output node they passed in.
	inputPos := make([]int, len(nodes))
	outputPos := make([]int, len(nodes))
	numInputs, numOutputs := 0, 0
	for i, node := range nodes {
		inputPos[i], outputPos[i] = -1, -1
		switch node.Type {
		case Input:
			inputPos[i] = numInputs
			numInputs++
		case Output:
			outputPos[i] = numOutputs
			numOutputs++
		}
	}

	// Collect the connections that actually contribute, counting each node's
	// degree as we go.
	type edge struct {
		from, to, connection int
	}
	edges := make([]edge, 0, len(connections))
	inDegree := make([]int, len(nodes))
	outDegree := make([]int, len(nodes))
	for i, connection := range connections {
		if !connection.Enabled {
			continue
		}
		from, ok := nodeIndex[connection.From]
		if !ok {
			continue
		}
		to, ok := nodeIndex[connection.To]
		if !ok {
			continue
		}
		edges = append(edges, edge{from: from, to: to, connection: i})
		inDegree[to]++
		outDegree[from]++
	}

	// Lay the edges out per node: start[i]:start[i+1] is node i's range.
	outStart := prefixSums(outDegree)
	inStart := prefixSums(inDegree)
	outTargets := make([]int, len(edges))
	inEdges := make([]int, len(edges))
	outCursor := make([]int, len(nodes))
	inCursor := make([]int, len(nodes))
	copy(outCursor, outStart)
	copy(inCursor, inStart)
	for i, e := range edges {
		outTargets[outCursor[e.from]] = e.to
		outCursor[e.from]++
		inEdges[inCursor[e.to]] = i
		inCursor[e.to]++
	}

	// topologicalOrder consumes inDegree, which is not needed afterwards.
	order, err := topologicalOrder(outStart, outTargets, inDegree)
	if err != nil {
		return nil, err
	}

	// position maps a node's index in nodes to its index in the evaluation order.
	position := make([]int, len(nodes))
	for pos, i := range order {
		position[i] = pos
	}

	net := &Network{
		nodes:          make([]compiledNode, len(order)),
		outputs:        make([]int, numOutputs),
		numInputs:      numInputs,
		numConnections: len(edges),
	}

	// One backing array for every node's inputs; each node takes a window of
	// it. Total appends can never exceed len(edges), so it never reallocates
	// and the windows handed out stay valid.
	allInputs := make([]weightedInput, 0, len(edges))

	for pos, i := range order {
		node := nodes[i]
		compiled := compiledNode{kind: node.Type, bias: node.Bias, inputPos: inputPos[i]}

		// Input and bias nodes are constant sources: they ignore their bias,
		// their activation and anything wired into them.
		if node.Type != Input && node.Type != Bias {
			activation := ActivationRegistry.Get(node.ActivationFn)
			if activation == nil {
				return nil, fmt.Errorf("%w: node %d uses %q", ErrUnknownActivation, node.ID, node.ActivationFn)
			}
			compiled.activation = activation

			from := len(allInputs)
			for k := inStart[i]; k < inStart[i+1]; k++ {
				e := edges[inEdges[k]]
				allInputs = append(allInputs, weightedInput{
					from:   position[e.from],
					weight: connections[e.connection].Weight,
				})
			}
			compiled.inputs = allInputs[from:len(allInputs):len(allInputs)]
		}

		net.nodes[pos] = compiled
		if outputPos[i] >= 0 {
			net.outputs[outputPos[i]] = pos
		}
	}

	size := len(net.nodes)
	net.scratch.New = func() any {
		buffer := make([]float64, size)
		return &buffer
	}
	return net, nil
}

// prefixSums turns per-node counts into start offsets, so that node i owns the
// range [out[i], out[i+1]).
func prefixSums(counts []int) []int {
	starts := make([]int, len(counts)+1)
	for i, count := range counts {
		starts[i+1] = starts[i] + count
	}
	return starts
}

// topologicalOrder returns an evaluation order in which every node appears
// after all the nodes feeding it, using Kahn's algorithm over the flat
// adjacency layout. It consumes inDegree.
func topologicalOrder(outStart, outTargets, inDegree []int) ([]int, error) {
	order := make([]int, 0, len(inDegree))
	queue := make([]int, 0, len(inDegree))
	for i := range inDegree {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		i := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		order = append(order, i)
		for k := outStart[i]; k < outStart[i+1]; k++ {
			to := outTargets[k]
			inDegree[to]--
			if inDegree[to] == 0 {
				queue = append(queue, to)
			}
		}
	}
	if len(order) != len(inDegree) {
		return nil, fmt.Errorf("%w and cannot be activated feed-forward", ErrCycle)
	}
	return order, nil
}

// NumInputs returns the number of input values Activate expects.
func (n *Network) NumInputs() int { return n.numInputs }

// NumOutputs returns the number of output values Activate produces.
func (n *Network) NumOutputs() int { return len(n.outputs) }

// NumNodes returns the number of nodes in the network.
func (n *Network) NumNodes() int { return len(n.nodes) }

// NumConnections returns the number of enabled connections in the network.
// Together with NumNodes this is what a complexity penalty is usually built on.
func (n *Network) NumConnections() int { return n.numConnections }

// Activate runs the input through the network and returns its output.
//
// Safe to call concurrently from multiple goroutines.
func (n *Network) Activate(input []float64) ([]float64, error) {
	output := make([]float64, len(n.outputs))
	if err := n.ActivateInto(input, output); err != nil {
		return nil, err
	}
	return output, nil
}

// ActivateInto is Activate writing into a caller-supplied output slice, which
// must be exactly NumOutputs long. Use it in a hot loop to avoid allocating an
// output slice per activation.
//
// Safe to call concurrently from multiple goroutines, as long as each has its
// own output slice.
func (n *Network) ActivateInto(input, output []float64) error {
	if len(input) != n.numInputs {
		return fmt.Errorf("%w: network expects %d inputs, got %d", ErrInputSize, n.numInputs, len(input))
	}
	if len(output) != len(n.outputs) {
		return fmt.Errorf("%w: network produces %d outputs, got a buffer of %d", ErrOutputSize, len(n.outputs), len(output))
	}

	buffer := n.scratch.Get().(*[]float64)
	defer n.scratch.Put(buffer)
	values := *buffer

	for i := range n.nodes {
		node := &n.nodes[i]
		switch node.kind {
		case Input:
			// Sensors pass their value through untouched.
			values[i] = input[node.inputPos]
		case Bias:
			values[i] = 1
		default:
			state := node.bias
			for _, in := range node.inputs {
				state += values[in.from] * in.weight
			}
			values[i] = node.activation(state)
		}
	}

	for k, i := range n.outputs {
		output[k] = values[i]
	}
	return nil
}
