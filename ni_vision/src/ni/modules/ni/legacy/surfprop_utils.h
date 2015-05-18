#ifndef _NI_LEGACY_SURFPROP_UTILS_H_
#define _NI_LEGACY_SURFPROP_UTILS_H_

#include <vector>

#include "ni/core/surface.h"

struct SurfProp;

namespace ni {

void SurfPropToVecSurfaces(const SurfProp &surf_prop, std::vector<ni::Surface> &surfaces);

void VecSurfacesToSurfProp(const std::vector<ni::Surface> &surfaces, SurfProp &surf_prop);

} // namespace ni

#endif // _NI_LEGACY_SURFPROP_UTILS_H_
