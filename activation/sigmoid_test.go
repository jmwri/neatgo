package activation

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"math"
	"reflect"
	"testing"
)

func TestSigmoid(t *testing.T) {
	fn, err := DefaultRegistry.Get("sigmoid")
	assert.NoError(t, err)
	assert.EqualValues(t, Sigmoid.Name(), fn.Name())
	assert.EqualValues(t, reflect.ValueOf(Sigmoid.fn), reflect.ValueOf(fn.fn))

	type testCase struct {
		input, expected float64
	}
	tests := []testCase{
		{
			input:    0,
			expected: 0.5,
		},
		{
			input:    math.Inf(-1),
			expected: 0,
		},
		{
			input:    math.Inf(1),
			expected: 1,
		},
		{
			input:    37,
			expected: 1,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(fmt.Sprint(test.input), func(t *testing.T) {
			actual := fn.Run(test.input)
			assert.Equal(t, test.expected, actual)
		})
	}
}
