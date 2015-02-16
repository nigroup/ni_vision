#ifndef _NI_CORE_BASE_BOUNDINGBOX_H_
#define _NI_CORE_BASE_BOUNDINGBOX_H_

#include "elm/core/typedefs_fwd.h"

namespace ni {

/**
 * @brief base bounding box class
 */
class base_BoundingBox
{
public:
    virtual ~base_BoundingBox();

    virtual float diagonal() const = 0;

    virtual cv::Mat1f centralPoint() const = 0;

protected:
    base_BoundingBox();
};

} // namespace ni

#endif // _NI_CORE_BASE_BOUNDINGBOX_H_
