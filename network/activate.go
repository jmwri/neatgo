package network

// Activate compiles the given nodes and connections and runs a single input
// through them.
//
// This is a convenience for one-off evaluations. Anything that activates the
// same network more than once should Compile it once and reuse the resulting
// *Network, which is both far faster and safe to share between goroutines.
func Activate(nodes []Node, connections []Connection, input []float64) ([]float64, error) {
	net, err := Compile(nodes, connections)
	if err != nil {
		return nil, err
	}
	return net.Activate(input)
}
