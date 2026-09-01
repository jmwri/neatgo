package neat_test

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A trained genome is the thing a run exists to produce, so it has to survive a
// round trip to disk. Genome is a plain tree of exported types, which means
// encoding/json handles it without any custom marshalling - but that is a
// promise worth holding the package to.
func TestGenome_SurvivesAJSONRoundTrip(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 40
	cfg.Seed = 7
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)

	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	answers := []float64{0, 1, 1, 0}

	// Train briefly so the genome has real structure: hidden nodes, disabled
	// connections and a multi-layer shape.
	pop, err = neat.Run(context.Background(), pop, func(_ context.Context, net *network.Network) (float64, error) {
		fitness := .0
		for i, input := range inputs {
			output, err := net.Activate(input)
			if err != nil {
				return 0, err
			}
			fitness += 1 - math.Pow(output[0]-answers[i], 2)
		}
		return fitness, nil
	}, neat.RunOptions{MaxGenerations: 40})
	require.NoError(t, err)

	original := pop.BestEverGenome
	require.NotZero(t, original.NumNodes())

	encoded, err := json.Marshal(original)
	require.NoError(t, err)

	var restored neat.Genome
	require.NoError(t, json.Unmarshal(encoded, &restored))

	assert.Equal(t, original.NumLayers(), restored.NumLayers())
	assert.Equal(t, original.NumNodes(), restored.NumNodes())
	assert.Equal(t, original.NumConnections(), restored.NumConnections())
	assert.Equal(t, original.Layers, restored.Layers)
	assert.Equal(t, original.Connections, restored.Connections)

	// And it still computes exactly the same thing.
	before, err := original.Compile()
	require.NoError(t, err)
	after, err := restored.Compile()
	require.NoError(t, err)
	for _, input := range inputs {
		want, err := before.Activate(input)
		require.NoError(t, err)
		got, err := after.Activate(input)
		require.NoError(t, err)
		assert.Equal(t, want, got, "input %v", input)
	}
}

// A hand-written genome loads too, so a network can be authored or generated
// outside the library.
func TestGenome_LoadsFromHandWrittenJSON(t *testing.T) {
	const doc = `{
		"Layers": [
			[
				{"ID": 1, "Type": "input", "Bias": 0, "ActivationFn": "no-activation"},
				{"ID": 2, "Type": "bias", "Bias": 0, "ActivationFn": "no-activation"}
			],
			[
				{"ID": 3, "Type": "output", "Bias": 0, "ActivationFn": "identity"}
			]
		],
		"Connections": [
			{"ID": 4, "From": 1, "To": 3, "Weight": 2, "Enabled": true},
			{"ID": 5, "From": 2, "To": 3, "Weight": 0.5, "Enabled": true}
		]
	}`

	var genome neat.Genome
	require.NoError(t, json.Unmarshal([]byte(doc), &genome))

	net, err := genome.Compile()
	require.NoError(t, err)
	assert.Equal(t, 1, net.NumInputs())
	assert.Equal(t, 1, net.NumOutputs())

	// input*2 + bias*0.5 = 3*2 + 1*0.5
	output, err := net.Activate([]float64{3})
	require.NoError(t, err)
	assert.Equal(t, []float64{6.5}, output)
}
