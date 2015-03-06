#ifndef _NI_CORE_BOUNDINGBOX2D_H_
#define _NI_CORE_BOUNDINGBOX2D_H_

#include "ni/core/base_boundingbox.h"

#include <opencv2/core/core.hpp>

#include "elm/core/pcl/typedefs_fwd.h"

namespace ni {

/**
 * @brief class for 2-dimensional bounding box
 * In image/pixel space increasing to the left and downwards
 */
class BoundingBox2D :
        public base_BoundingBox,
        public cv::Rect2i
{
public:
    BoundingBox2D();

    BoundingBox2D(const cv::Rect2i &rect);

    float diagonal() const;

    cv::Mat1f centralPoint() const;

protected:

    // members
};

} // namespace ni

#endif // _NI_CORE_BOUNDINGBOX2D_H_
