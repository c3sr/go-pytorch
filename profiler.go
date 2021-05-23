package pytorch

// #include "cbits/predictor.hpp"
// #include <stdlib.h>
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.Torch_ProfilingRead(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}
