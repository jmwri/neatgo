package activation

import (
	"fmt"
	"github.com/jmwri/neatgo/util"
	"github.com/stretchr/testify/assert"
	"reflect"
	"testing"
)

func TestNone(t *testing.T) {
	fn, err := DefaultRegistry.Get("none")
	assert.NoError(t, err)
	assert.EqualValues(t, None.Name(), fn.Name())
	assert.EqualValues(t, reflect.ValueOf(None.fn), reflect.ValueOf(fn.fn))

	type testCase struct {
		input, expected float64
	}
	tests := make([]testCase, 0)
	for i := 0; i <= 10; i++ {
		v := util.FloatBetween(-10, 10)
		tests = append(tests, testCase{
			input:    v,
			expected: v,
		})
	}

	for _, test := range tests {
		test := test
		t.Run(fmt.Sprint(test.input), func(t *testing.T) {
			actual := fn.Run(test.input)
			assert.Equal(t, test.expected, actual)
		})
	}
}
