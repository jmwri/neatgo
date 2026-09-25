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
// of goroutines. A recurrent one keeps no state of its own either: what it
// remembers between activations lives in a Memory, which the caller owns and
// which is what makes it safe to run the same recurrent network on several
// goroutines at once.
type Network struct {
	nodes          []compiledNode // in evaluation order
	outputs        []int          // index into nodes for each output position
	numInputs      int
	numConnections int
	recurrent      bool
	scratch        sync.Pool
}

type compiledNode struct {
	kind       nodeKind
	bias       float64
	activation ActivationFunction
	// activationName is the registry name the activation was looked up by, kept
	// so that Program can describe the network to something that cannot call the
	// function itself.
	activationName ActivationFunctionName
	inputs         []weightedInput
	// remembered are the incoming connections that do not run forwards in the
	// evaluation order. They read the value their source held at the end of the
	// previous activation, which is what lets a recurrent network carry anything
	// from one step to the next.
	remembered []weightedInput
	// inputPos is the position in the input slice, for Input nodes only.
	inputPos int
}

// nodeKind is what Step switches on for every node of every activation. It is
// a small integer rather than the NodeType string so that the dispatch is one
// byte compare and not a string comparison per node.
type nodeKind uint8

const (
	kindComputed nodeKind = iota // hidden and output nodes
	kindInput
	kindBias
)

func kindOf(t NodeType) nodeKind {
	switch t {
	case Input:
		return kindInput
	case Bias:
		return kindBias
	}
	return kindComputed
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
// Connections that are disabled, that reference a node not in nodes, or that
// lead into an input or bias node are dropped. Compile fails if the remaining
// connections contain a cycle, if a node uses an unregistered activation
// function or an unknown type, or if two nodes share an ID.
//
// The graph is held in flat arrays with per-node offsets rather than a slice
// of slices. A slice per node would allocate once per node and again as each
// grew; this allocates a fixed handful of times no matter how large the
// network is, which matters because a population compiles every genome afresh
// each generation.
func Compile(nodes []Node, connections []Connection) (*Network, error) {
	return compile(nodes, connections, false)
}

// CompileRecurrent builds a network that may contain loops.
//
// Nodes are evaluated in the order they are given, which for a genome is layer
// order with each layer in the order its nodes were added. A connection from a
// node earlier in that order to one later behaves exactly as it does in a
// feed-forward network; any other - to a node earlier in the order, or to
// itself - reads the value its source held at the end of the previous
// activation. So there is no cycle to resolve within a single pass, and a genome
// with no backward connections compiles to the same network either way.
//
// What it buys is memory: the network's answer can depend on what it has already
// seen, not only on what it is being shown now. What it costs is that activation
// is no longer a pure function of the input, so a Memory has to be carried
// between steps and reset between episodes.
func CompileRecurrent(nodes []Node, connections []Connection) (*Network, error) {
	return compile(nodes, connections, true)
}

func compile(nodes []Node, connections []Connection, recurrent bool) (*Network, error) {
	nodeIndex, err := newNodeIndex(nodes)
	if err != nil {
		return nil, err
	}

	// Every per-node integer table is carved out of one allocation. Compile
	// runs once per genome per generation, and what it costs is dominated by
	// the garbage it leaves behind rather than the work it does.
	n := len(nodes)
	tables := newIntTables(10*n + 2)

	// Input and output positions follow the order the nodes were given in, not
	// the evaluation order, so that callers can rely on output[k] being the
	// k-th output node they passed in.
	inputPos := tables.take(n)
	outputPos := tables.take(n)
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
		case Hidden, Bias:
		default:
			// A genome written by hand with a misspelt type would otherwise
			// quietly become a hidden node, and the caller's inputs or
			// outputs would no longer line up with the nodes they meant.
			return nil, fmt.Errorf("%w: node %d has type %q", ErrUnknownNodeType, node.ID, node.Type)
		}
	}

	// Collect the connections that actually contribute, counting each node's
	// degree as we go.
	type edge struct {
		from, to, connection int
	}
	edges := make([]edge, 0, len(connections))
	inDegree := tables.take(n)
	outDegree := tables.take(n)
	for i, connection := range connections {
		if !connection.Enabled {
			continue
		}
		from, ok := nodeIndex.find(connection.From)
		if !ok {
			continue
		}
		to, ok := nodeIndex.find(connection.To)
		if !ok {
			continue
		}
		if kind := nodes[to].Type; kind == Input || kind == Bias {
			// A connection into a constant source has no effect on the
			// network, so it must not count as a connection or take part in
			// the cycle check: an edge that is never evaluated cannot close
			// a loop.
			continue
		}
		edges = append(edges, edge{from: from, to: to, connection: i})
		inDegree[to]++
		outDegree[from]++
	}

	// Lay the edges out per node: start[i]:start[i+1] is node i's range.
	outStart := prefixSums(outDegree, tables.take(n+1))
	inStart := prefixSums(inDegree, tables.take(n+1))
	edgeTables := newIntTables(2 * len(edges))
	outTargets := edgeTables.take(len(edges))
	inEdges := edgeTables.take(len(edges))
	outCursor := tables.take(n)
	inCursor := tables.take(n)
	copy(outCursor, outStart)
	copy(inCursor, inStart)
	for i, e := range edges {
		outTargets[outCursor[e.from]] = e.to
		outCursor[e.from]++
		inEdges[inCursor[e.to]] = i
		inCursor[e.to]++
	}

	// A recurrent network is evaluated in the order it was given, and decides
	// which connections run forwards from that. A feed-forward one has to be
	// sorted, and a cycle is an error rather than a memory.
	order := tables.take(n)
	if recurrent {
		for i := range order {
			order[i] = i
		}
	} else {
		// topologicalOrder consumes inDegree, which is not needed afterwards,
		// and uses outCursor, which is not either, as its work queue.
		if err := topologicalOrder(outStart, outTargets, inDegree, order, outCursor); err != nil {
			return nil, err
		}
	}

	// position maps a node's index in nodes to its index in the evaluation order.
	position := tables.take(n)
	for pos, i := range order {
		position[i] = pos
	}

	net := &Network{
		nodes:          make([]compiledNode, len(order)),
		outputs:        make([]int, numOutputs),
		numInputs:      numInputs,
		numConnections: len(edges),
		recurrent:      recurrent,
	}

	// One backing array for every node's inputs, forward and remembered
	// alike; each node takes two adjacent windows of it. Every edge lands in
	// exactly one window, so the total never exceeds len(edges): the array
	// never reallocates and the windows handed out stay valid.
	all := make([]weightedInput, 0, len(edges))

	for pos, i := range order {
		node := nodes[i]
		compiled := compiledNode{kind: kindOf(node.Type), bias: node.Bias, inputPos: inputPos[i]}

		// Input and bias nodes are constant sources: they ignore their bias,
		// their activation and anything wired into them.
		if node.Type != Input && node.Type != Bias {
			activation := ActivationRegistry.Get(node.ActivationFn)
			if activation == nil {
				return nil, fmt.Errorf("%w: node %d uses %q", ErrUnknownActivation, node.ID, node.ActivationFn)
			}
			compiled.activation = activation
			compiled.activationName = node.ActivationFn

			// pos is this node's own place in the order; a source not
			// strictly before it has not been computed yet this pass, so it
			// is read from memory instead. Two passes over the node's edges,
			// forward ones first, keep each kind contiguous.
			start := len(all)
			for k := inStart[i]; k < inStart[i+1]; k++ {
				e := edges[inEdges[k]]
				if position[e.from] < pos {
					all = append(all, weightedInput{from: position[e.from], weight: connections[e.connection].Weight})
				}
			}
			split := len(all)
			for k := inStart[i]; k < inStart[i+1]; k++ {
				e := edges[inEdges[k]]
				if position[e.from] >= pos {
					all = append(all, weightedInput{from: position[e.from], weight: connections[e.connection].Weight})
				}
			}
			compiled.inputs = all[start:split:split]
			compiled.remembered = all[split:len(all):len(all)]
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

// nodeIndex maps a node ID to its position in the nodes slice.
//
// IDs come from a sequential counter, so within one genome they are usually
// packed into a range not much wider than the genome itself, and a slice
// indexed by ID beats hashing. When they are spread out - a small genome late
// in a long run - a map is used instead, so the slice can never be large.
type nodeIndex struct {
	dense  []int // position by (ID - base), or -1
	base   int
	sparse map[int]int
}

func newNodeIndex(nodes []Node) (nodeIndex, error) {
	var index nodeIndex
	if len(nodes) == 0 {
		return index, nil
	}
	lo, hi := nodes[0].ID, nodes[0].ID
	for _, node := range nodes[1:] {
		lo = min(lo, node.ID)
		hi = max(hi, node.ID)
	}
	if span := hi - lo + 1; span <= 4*len(nodes)+64 {
		index.base = lo
		index.dense = make([]int, span)
		for i := range index.dense {
			index.dense[i] = -1
		}
		for i, node := range nodes {
			at := node.ID - lo
			if index.dense[at] >= 0 {
				return index, fmt.Errorf("%w: %d", ErrDuplicateNode, node.ID)
			}
			index.dense[at] = i
		}
		return index, nil
	}
	index.sparse = make(map[int]int, len(nodes))
	for i, node := range nodes {
		if _, duplicate := index.sparse[node.ID]; duplicate {
			return index, fmt.Errorf("%w: %d", ErrDuplicateNode, node.ID)
		}
		index.sparse[node.ID] = i
	}
	return index, nil
}

func (x nodeIndex) find(id int) (int, bool) {
	if x.dense != nil {
		at := id - x.base
		if at < 0 || at >= len(x.dense) || x.dense[at] < 0 {
			return 0, false
		}
		return x.dense[at], true
	}
	i, ok := x.sparse[id]
	return i, ok
}

// intTables hands out fixed-size integer slices from one backing array, so
// that the dozen small tables compile needs cost one allocation between them.
type intTables struct {
	buf []int
}

func newIntTables(size int) intTables {
	return intTables{buf: make([]int, size)}
}

// take returns the next size ints, zeroed, with no spare capacity so that an
// append can never spill into the table after it.
func (t *intTables) take(size int) []int {
	s := t.buf[:size:size]
	t.buf = t.buf[size:]
	return s
}

// prefixSums turns per-node counts into start offsets, so that node i owns the
// range [out[i], out[i+1]). starts must be one longer than counts.
func prefixSums(counts, starts []int) []int {
	starts[0] = 0
	for i, count := range counts {
		starts[i+1] = starts[i] + count
	}
	return starts
}

// topologicalOrder fills order with an evaluation order in which every node
// appears after all the nodes feeding it, using Kahn's algorithm over the flat
// adjacency layout. It consumes inDegree and uses queue, which must be as long
// as order, as scratch space.
func topologicalOrder(outStart, outTargets, inDegree, order, queue []int) error {
	queue = queue[:0]
	for i := range inDegree {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	// Every node is queued exactly once, so the queue can never outgrow its
	// backing array and order fills in lockstep with the nodes leaving it.
	sorted := 0
	for len(queue) > 0 {
		i := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		order[sorted] = i
		sorted++
		for k := outStart[i]; k < outStart[i+1]; k++ {
			to := outTargets[k]
			inDegree[to]--
			if inDegree[to] == 0 {
				queue = append(queue, to)
			}
		}
	}
	if sorted != len(inDegree) {
		return fmt.Errorf("%w and cannot be activated feed-forward", ErrCycle)
	}
	return nil
}

// NumInputs returns the number of input values Activate expects.
func (n *Network) NumInputs() int { return n.numInputs }

// NumOutputs returns the number of output values Activate produces.
func (n *Network) NumOutputs() int { return len(n.outputs) }

// NumNodes returns the number of nodes in the network.
func (n *Network) NumNodes() int { return len(n.nodes) }

// IsRecurrent reports whether the network was compiled with CompileRecurrent
// and so needs a Memory to activate.
func (n *Network) IsRecurrent() bool { return n.recurrent }

// Memory is what a recurrent network carries from one activation to the next.
//
// It belongs to whoever is running the network rather than to the network
// itself, so that one compiled network can be run on many goroutines at once,
// each with its own. Reset it between episodes: a network that starts a game
// still remembering the end of the last one is being asked a question about a
// board that no longer exists.
type Memory struct {
	previous []float64
}

// NewMemory returns memory sized for this network. Feed-forward networks have
// nothing to remember, but one is harmless.
func (n *Network) NewMemory() *Memory {
	return &Memory{previous: make([]float64, len(n.nodes))}
}

// Reset forgets everything, returning the memory to the state it started in.
func (m *Memory) Reset() {
	for i := range m.previous {
		m.previous[i] = 0
	}
}

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
// own output slice. A recurrent network has to be run with Step instead, since
// there is nowhere here to keep what it remembers.
func (n *Network) ActivateInto(input, output []float64) error {
	if n.recurrent {
		return ErrNeedsMemory
	}
	return n.Step(nil, input, output)
}

// Step runs the network for one activation, reading what it remembered from mem
// and writing back what it will remember next time.
//
// mem may be nil for a feed-forward network, which remembers nothing. For a
// recurrent one it is required, and must have come from that network's NewMemory.
//
// Safe to call concurrently from multiple goroutines, as long as each has its
// own memory and output slice.
func (n *Network) Step(mem *Memory, input, output []float64) error {
	if n.recurrent && mem == nil {
		return ErrNeedsMemory
	}
	if mem != nil && len(mem.previous) != len(n.nodes) {
		return fmt.Errorf("%w: network has %d nodes, memory holds %d", ErrMemorySize, len(n.nodes), len(mem.previous))
	}
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
		case kindInput:
			// Sensors pass their value through untouched.
			values[i] = input[node.inputPos]
		case kindBias:
			values[i] = 1
		default:
			state := node.bias
			for _, in := range node.inputs {
				state += values[in.from] * in.weight
			}
			for _, in := range node.remembered {
				state += mem.previous[in.from] * in.weight
			}
			values[i] = node.activation(state)
		}
	}

	if mem != nil {
		copy(mem.previous, values)
	}

	for k, i := range n.outputs {
		output[k] = values[i]
	}
	return nil
}
