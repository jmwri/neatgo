//go:build !windows && !linux

package gpu

import "errors"

// openLibrary reports that CUDA does not exist here: there is no NVIDIA driver
// for this platform.
func openLibrary() (uintptr, error) {
	return 0, errors.New("CUDA is not supported on this platform")
}
