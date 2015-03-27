#ifndef _NI_CORE_BOUNDINGBOX3D_H_
#define _NI_CORE_BOUNDINGBOX3D_H_

#include "ni/core/base_boundingbox.h"

#include <opencv2/core/core.hpp>

#include "elm/core/pcl/typedefs_fwd.h"

namespace ni {

/**
 * @brief class for 3-dimensional bounding box
 */
class BoundingBox3D :
        public base_BoundingBox
{
public:
    BoundingBox3D();

    BoundingBox3D(elm::CloudXYZPtr &cld);

    float diagonal() const;

    cv::Mat1f centralPoint() const;

    float volume() const;

    cv::Mat1f cubeVertices() const;

    void cubeVertices(const cv::Mat1f& cube);

protected:

    // members
    cv::Mat1f cog_; ///< center of gravity

    cv::Mat1f box_xy_;    ///< bounding box in xy plane
    cv::Mat1f box_yz_;    ///< bounding box in yz plane
};

} // namespace ni

#endif // _NI_CORE_BOUNDINGBOX3D_H_
