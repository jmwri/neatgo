package gpu

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jmwri/neatgo/v2/network"
)

func openOrSkip(t *testing.T) *Device {
	t.Helper()
	if !Available() {
		t.Skip("no CUDA device")
	}
	d, err := Open(0)
	require.NoError(t, err)
	t.Cleanup(func() { d.Close() })
	return d
}

// close reports whether the GPU's float32 answer is the CPU's within the
// precision the GPU is documented to have.
func close(want, got float64) bool {
	return math.Abs(want-got) <= 2e-4+1e-4*math.Abs(want)
}

// A single hidden node with the activation under test, between an input and an
// output that just pass the value on.
func activationNet(t *testing.T, name network.ActivationFunctionName) *network.Network {
	t.Helper()
	nodes := []network.Node{
		network.NewNode(0, network.Input, 0, network.NoActivation),
		network.NewNode(1, network.Hidden, .1, name),
		network.NewNode(2, network.Output, 0, network.Identity),
	}
	conns := []network.Connection{
		{ID: 0, From: 0, To: 1, Weight: 1.3, Enabled: true},
		{ID: 1, From: 1, To: 2, Weight: 1, Enabled: true},
	}
	net, err := network.Compile(nodes, conns)
	require.NoError(t, err)
	return net
}

func TestEveryActivationMatchesTheCPU(t *testing.T) {
	d := openOrSkip(t)

	var inputs [][]float64
	for x := -6.0; x <= 6.0; x += .0625 {
		inputs = append(inputs, []float64{x})
	}

	for name := range activationOps {
		t.Run(string(name), func(t *testing.T) {
			net := activationNet(t, name)
			want, err := network.ActivateBatch([]*network.Network{net}, inputs, 1)
			require.NoError(t, err)
			before := d.launched
			got, err := d.ActivateBatch([]*network.Network{net}, inputs)
			require.NoError(t, err)
			require.Greater(t, d.launched, before, "batch did not run on the GPU")

			for s := range inputs {
				w, g := want.Net(0).Sample(s)[0], got.Net(0).Sample(s)[0]
				assert.Truef(t, close(w, g), "x=%v: cpu %v gpu %v", inputs[s][0], w, g)
			}
		})
	}
}

func TestBiasAndDisabledAndSharedInputs(t *testing.T) {
	d := openOrSkip(t)
	nodes := []network.Node{
		network.NewNode(0, network.Input, 0, network.NoActivation),
		network.NewNode(1, network.Input, 0, network.NoActivation),
		network.NewNode(2, network.Bias, 0, network.NoActivation),
		network.NewNode(3, network.Hidden, -.2, network.Tanh),
		network.NewNode(4, network.Output, .3, network.Sigmoid),
		network.NewNode(5, network.Output, -.4, network.Relu),
	}
	conns := []network.Connection{
		{ID: 0, From: 1, To: 3, Weight: .7, Enabled: true}, // input order != node order
		{ID: 1, From: 0, To: 3, Weight: -1.2, Enabled: true},
		{ID: 2, From: 2, To: 3, Weight: .5, Enabled: true},
		{ID: 3, From: 3, To: 4, Weight: 2, Enabled: true},
		{ID: 4, From: 0, To: 5, Weight: 3, Enabled: true},
		{ID: 5, From: 1, To: 5, Weight: 9, Enabled: false},
		{ID: 6, From: 3, To: 5, Weight: -1, Enabled: true},
	}
	net, err := network.Compile(nodes, conns)
	require.NoError(t, err)

	rng := rand.New(rand.NewPCG(1, 2))
	inputs := make([][]float64, 200)
	for i := range inputs {
		inputs[i] = []float64{rng.Float64()*4 - 2, rng.Float64()*4 - 2}
	}
	want, err := network.ActivateBatch([]*network.Network{net}, inputs, 1)
	require.NoError(t, err)
	got, err := d.ActivateBatch([]*network.Network{net}, inputs)
	require.NoError(t, err)
	for s := range inputs {
		for o := 0; o < 2; o++ {
			assert.Truef(t, close(want.Net(0).Sample(s)[o], got.Net(0).Sample(s)[o]), "sample %d output %d", s, o)
		}
	}
}

// randomNet builds a layered network of random size and weights, so that a
// population of them differs in node count, edge count and activations.
func randomNet(t *testing.T, rng *rand.Rand, numIn, numOut int) *network.Network {
	t.Helper()
	acts := []network.ActivationFunctionName{network.Sigmoid, network.Tanh, network.Relu, network.Gauss, network.Sin, network.Elu}
	var nodes []network.Node
	id := 0
	add := func(kind network.NodeType, act network.ActivationFunctionName) int {
		nodes = append(nodes, network.NewNode(id, kind, rng.NormFloat64()*.5, act))
		id++
		return id - 1
	}
	var inIDs, hidIDs, outIDs []int
	for i := 0; i < numIn; i++ {
		inIDs = append(inIDs, add(network.Input, network.NoActivation))
	}
	for i, n := 0, rng.IntN(12); i < n; i++ {
		hidIDs = append(hidIDs, add(network.Hidden, acts[rng.IntN(len(acts))]))
	}
	for i := 0; i < numOut; i++ {
		outIDs = append(outIDs, add(network.Output, network.Sigmoid))
	}
	var conns []network.Connection
	link := func(from, to int) {
		if rng.Float64() < .6 {
			conns = append(conns, network.Connection{ID: len(conns), From: from, To: to, Weight: rng.NormFloat64(), Enabled: rng.Float64() < .9})
		}
	}
	for _, f := range inIDs {
		for _, h := range hidIDs {
			link(f, h)
		}
		for _, o := range outIDs {
			link(f, o)
		}
	}
	for i, h := range hidIDs {
		for _, o := range outIDs {
			link(h, o)
		}
		for _, h2 := range hidIDs[i+1:] {
			link(h, h2)
		}
	}
	net, err := network.Compile(nodes, conns)
	require.NoError(t, err)
	return net
}

func randomBatch(t *testing.T, nets, samples, numIn, numOut int) ([]*network.Network, [][]float64) {
	rng := rand.New(rand.NewPCG(7, 11))
	ns := make([]*network.Network, nets)
	for i := range ns {
		ns[i] = randomNet(t, rng, numIn, numOut)
	}
	inputs := make([][]float64, samples)
	for i := range inputs {
		inputs[i] = make([]float64, numIn)
		for j := range inputs[i] {
			inputs[i][j] = rng.Float64()*2 - 1
		}
	}
	return ns, inputs
}

func compare(t *testing.T, want, got network.BatchOutputs) {
	t.Helper()
	require.Equal(t, want.NumNets, got.NumNets)
	require.Equal(t, want.NumSamples, got.NumSamples)
	require.Equal(t, want.NumOutputs, got.NumOutputs)
	bad := 0
	for i := range want.Data {
		if !close(want.Data[i], got.Data[i]) {
			bad++
			if bad <= 5 {
				t.Errorf("index %d: cpu %v gpu %v", i, want.Data[i], got.Data[i])
			}
		}
	}
	assert.Zero(t, bad, "mismatched values")
}

func TestRandomPopulationMatchesTheCPU(t *testing.T) {
	d := openOrSkip(t)
	nets, inputs := randomBatch(t, 60, 500, 5, 3)
	want, err := network.ActivateBatch(nets, inputs, 0)
	require.NoError(t, err)
	got, err := d.ActivateBatch(nets, inputs)
	require.NoError(t, err)
	require.NotZero(t, d.launched, "batch did not run on the GPU")
	compare(t, want, got)
}

func TestSplitLaunchesMatchTheCPU(t *testing.T) {
	d := openOrSkip(t)
	nets, inputs := randomBatch(t, 23, 97, 4, 2)
	want, err := network.ActivateBatch(nets, inputs, 0)
	require.NoError(t, err)

	// A budget of a few hundred floats forces both the network and the sample
	// dimension to be chunked, including ragged final chunks.
	for _, limit := range []uintptr{4 * 40, 4 * 300, 4 * 5000} {
		d.scratchLimit = limit
		before := d.launched
		got, err := d.ActivateBatch(nets, inputs)
		require.NoError(t, err, "limit %d", limit)
		require.Greater(t, d.launched, before+1, "limit %d should have split the launch", limit)
		compare(t, want, got)
	}
}

func TestCustomActivationFallsBackToTheCPU(t *testing.T) {
	d := openOrSkip(t)
	const name network.ActivationFunctionName = "test-triple"
	network.ActivationRegistry.Set(name, func(x float64) float64 { return 3 * x })
	t.Cleanup(func() { network.ActivationRegistry.Set(name, nil) })

	custom := activationNet(t, name)
	plain := activationNet(t, network.Tanh)
	inputs := [][]float64{{.5}, {-1}, {2}}

	nets := []*network.Network{plain, custom, plain}
	want, err := network.ActivateBatch(nets, inputs, 1)
	require.NoError(t, err)
	got, err := d.ActivateBatch(nets, inputs)
	require.NoError(t, err)
	compare(t, want, got)
	// The custom network must have used its own function, not a built-in.
	assert.InDelta(t, 3*(1.3*.5+.1), got.Net(1).Sample(0)[0], 1e-9)
}

func TestReplacedBuiltinIsNotRunOnTheGPU(t *testing.T) {
	d := openOrSkip(t)
	original := network.ActivationRegistry.Get(network.Relu)
	network.ActivationRegistry.Set(network.Relu, func(x float64) float64 { return -x })
	t.Cleanup(func() { network.ActivationRegistry.Set(network.Relu, original) })

	net := activationNet(t, network.Relu)
	got, err := d.ActivateBatch([]*network.Network{net}, [][]float64{{1}})
	require.NoError(t, err)
	assert.InDelta(t, -(1.3 + .1), got.Net(0).Sample(0)[0], 1e-9)
}

func TestRejectsWhatABatchCannotRun(t *testing.T) {
	d := openOrSkip(t)
	nodes := []network.Node{
		network.NewNode(0, network.Input, 0, network.NoActivation),
		network.NewNode(1, network.Output, 0, network.Identity),
	}
	conns := []network.Connection{{ID: 0, From: 0, To: 1, Weight: 1, Enabled: true}, {ID: 1, From: 1, To: 1, Weight: .5, Enabled: true}}
	recurrent, err := network.CompileRecurrent(nodes, conns)
	require.NoError(t, err)

	_, err = d.ActivateBatch([]*network.Network{recurrent}, [][]float64{{1}})
	assert.ErrorIs(t, err, network.ErrNotFeedForward)

	plain := activationNet(t, network.Tanh)
	_, err = d.ActivateBatch([]*network.Network{plain}, [][]float64{{1, 2}})
	assert.ErrorIs(t, err, network.ErrInputSize)
}

func TestEmptyBatches(t *testing.T) {
	d := openOrSkip(t)
	out, err := d.ActivateBatch(nil, [][]float64{{1}})
	require.NoError(t, err)
	assert.Empty(t, out.Data)

	out, err = d.ActivateBatch([]*network.Network{activationNet(t, network.Tanh)}, nil)
	require.NoError(t, err)
	assert.Empty(t, out.Data)
}

func TestUnavailableIsDistinguishable(t *testing.T) {
	_, err := Open(9999)
	if err == nil {
		t.Fatal("opening device 9999 succeeded")
	}
	assert.True(t, errors.Is(err, ErrUnavailable))
}
