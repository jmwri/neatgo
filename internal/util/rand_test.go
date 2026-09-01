package util_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jmwri/neatgo/v2/internal/util"
	"github.com/stretchr/testify/assert"
)

func testRand() *rand.Rand { return rand.New(rand.NewPCG(1, 2)) }

func TestFloatBetween(t *testing.T) {
	rng := testRand()
	tests := []struct{ min, max float64 }{
		{0, 100}, {0, 1}, {5, 6}, {5, 5.2}, {9999, 10000}, {-10, 0}, {-1, 1},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%v", test), func(t *testing.T) {
			for i := 0; i < 100; i++ {
				actual := util.FloatBetween(rng, test.min, test.max)
				assert.GreaterOrEqual(t, actual, test.min)
				assert.LessOrEqual(t, actual, test.max)
			}
		})
	}
}

func TestChance(t *testing.T) {
	rng := testRand()

	// Zero and one must be exact, not merely very likely, so that a mutation
	// rate of 0 is genuinely off and 1 is genuinely always.
	for i := 0; i < 100; i++ {
		assert.False(t, util.Chance(rng, 0))
		assert.False(t, util.Chance(rng, -1))
		assert.True(t, util.Chance(rng, 1))
		assert.True(t, util.Chance(rng, 2))
	}

	hits := 0
	const trials = 20000
	for i := 0; i < trials; i++ {
		if util.Chance(rng, .25) {
			hits++
		}
	}
	assert.InDelta(t, .25, float64(hits)/trials, .02)
}

func TestGaussian(t *testing.T) {
	rng := testRand()
	const samples = 50000
	sum, sumSq := 0.0, 0.0
	for i := 0; i < samples; i++ {
		v := util.Gaussian(rng)
		assert.False(t, math.IsNaN(v))
		assert.False(t, math.IsInf(v, 0))
		sum += v
		sumSq += v * v
	}
	mean := sum / samples
	variance := sumSq/samples - mean*mean
	assert.InDelta(t, 0, mean, .05, "standard normal should be centred on zero")
	assert.InDelta(t, 1, variance, .05, "standard normal should have unit variance")
}

func TestClamp(t *testing.T) {
	assert.Equal(t, 5.0, util.Clamp(10, -5, 5))
	assert.Equal(t, -5.0, util.Clamp(-10, -5, 5))
	assert.Equal(t, 1.5, util.Clamp(1.5, -5, 5))
	assert.Equal(t, 5.0, util.Clamp(5, -5, 5))
}

// The same seed must produce the same sequence, which is what makes a whole
// evolutionary run reproducible.
func TestRandIsDeterministicForASeed(t *testing.T) {
	a, b := testRand(), testRand()
	for i := 0; i < 1000; i++ {
		assert.Equal(t, util.FloatBetween(a, -1, 1), util.FloatBetween(b, -1, 1))
	}
}
