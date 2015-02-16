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

int Surface::lastSeenCount() const
{
    return last_seen_count_;
}

void Surface::lastSeenCount(bool is_last_seen)
{
    if(is_last_seen) {

        last_seen_count_ = 0;
    }
    else {
        last_seen_count_++;
    }
}
