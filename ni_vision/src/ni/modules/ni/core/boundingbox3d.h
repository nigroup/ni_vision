#ifndef _NI_CORE_BOUNDINGBOX3D_H_
#define _NI_CORE_BOUNDINGBOX3D_H_

#include "ni/core/base_boundingbox.h"

#include "elm/core/pcl/typedefs_fwd.h"

#include <opencv2/core/core.hpp>

namespace ni {

/**
 * @brief class for 2-dimensional bounding box
 */
class BoundingBox3D :
        public base_BoundingBox
{
public:
    BoundingBox3D();

    BoundingBox3D(elm::CloudXYZPtr &cld);

    float diagonal() const;

    cv::Mat1f centralPoint() const;

protected:

    // members
    cv::Mat1f cog_; ///< center of gravity
};

} // namespace ni

#endif // _NI_CORE_BOUNDINGBOX3D_H_
