//go:build linux

package gpu

import (
	"errors"

	"github.com/ebitengine/purego"
)

// openLibrary loads the CUDA driver, which ships with the NVIDIA display driver
// rather than with any toolkit.
func openLibrary() (uintptr, error) {
	var errs []error
	for _, name := range []string{"libcuda.so.1", "libcuda.so"} {
		handle, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			return handle, nil
		}
		errs = append(errs, err)
	}
	return 0, errors.Join(errs...)
}
