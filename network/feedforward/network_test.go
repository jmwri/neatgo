package feedforward

import (
	"context"
	"github.com/jmwri/neatgo/activation"
	"github.com/stretchr/testify/assert"
	"os"
	"testing"
)

func TestNetwork_Activate(t *testing.T) {
	net := NewNetwork(
		[]Node{
			{
				id:    1,
				actFn: activation.None,
				bias:  0,
			},
		},
		[]Node{
			{
				id:    2,
				actFn: activation.None,
				bias:  0,
			},
		},
		[]Node{
			{
				id:    3,
				actFn: activation.None,
				bias:  0,
			},
		},
		[]Connection{
			{
				id:      1,
				from:    1,
				to:      2,
				weight:  1,
				enabled: true,
			},
			{
				id:      2,
				from:    2,
				to:      3,
				weight:  1,
				enabled: true,
			},
		},
	)

	/**
	node(5 + 0 = 5) -> conn(5 * 1 = 1) -> node(5 + 0 = 5) -> conn(5 * 1 = 1) -> node(5 + 0 = 5)
	*/

	err := net.Activate(context.TODO(), 5)
	assert.NoError(t, err)

	_, err = net.Stats().Printf(os.Stdout)
	assert.NoError(t, err)

	output := net.Output()
	assert.Equal(t, []float64{5}, output)
}
