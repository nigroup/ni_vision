#include "ni/core/surface.h"

using namespace std;
using namespace ni;

Surface::Surface()
{
}

int Surface::pixelCount()
{
    return static_cast<int>(pixel_indices_.size());
}

void Surface::pixelIndices(const VecI &v)
{
    pixel_indices_ = v;
}

VecI Surface::pixelIndices() const
{
    return pixel_indices_;
}
