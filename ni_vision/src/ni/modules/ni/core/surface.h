#ifndef _NI_CORE_SURFACE_H_
#define _NI_CORE_SURFACE_H_

#include <opencv2/core/core.hpp>

#include "elm/core/cv/typedefs_fwd.h"
#include "ni/core/stl/typedefs.h"

namespace ni {

/**
 * @brief class for encapsulating Surface representation and attributes
 */
class Surface
{
public:
    /**
     * @brief count no. of pixels belonging to this surface
     * @return count of pixels for this surface
     */
    int pixelCount() const;

    /**
     * @brief overwrite count for no. of pixels belonging to this surface
     * @param count of pixels for this surface
     * @todo handle conflict with size of pixelIndices() vector
     */
    void overwritePixelCount(int count);

    /**
     * @brief Set pixel indicies belonging to this surface
     *
     * Involves deep copy
     *
     * @param v new vector of indices
     */
    void pixelIndices(const VecI &v);

    /**
     * @brief Get pixel indicies belonging to this surface
     *
     * Involves deep copy
     *
     * @return pixel indicies
     */
    VecI pixelIndices() const;

    /**
     * @brief get count of frames since this surface was last seen
     * @return frame count since last seen - zero indicates seen in most recent frame
     */
    int lastSeenCount() const;

    /**
     * @brief update last seen count
     * @return flag indicating it was last seen in most recent frame
     */
    void lastSeenCount(bool is_last_seen);

    /**
     * @brief Set surface id
     * @param new_id
     */
    void id(int new_id);

    /**
     * @brief get surface id
     * @return surface id
     */
    int id() const;

    /**
     * @brief get color histogram
     * @return color histogram (row matrix)
     */
    cv::Mat1f colorHistogram() const;

    /**
     * @brief set color histogram
     * @param hist new color histogram
     */
    void colorHistogram(const cv::Mat1f &hist);

    void diagonal(float d);

    float diagonal() const;

    void cubeCenter(const cv::Matx13f& c);

    cv::Matx13f cubeCenter() const;

    float distance(const Surface &s) const;

    /**
     * @brief rect
     * @return
     * @todo test
     */
    cv::Rect2i rect() const;

    /**
     * @brief rect
     * @todo test
     */
    void rect(const cv::Rect2i &rect);

    /**
     * @brief Default Constructor
     */
    Surface();

protected:

    // members
    VecI pixel_indices_;    ///< indicies to pixels belonging to this surface in flattened image
    int id_;                ///< surface identifier
    int last_seen_count_;   ///< count for no. of frames since last tracked
    int pixel_count_;       ///< cache for pixel count
    cv::Mat1f color_hist_;  ///< color histogram

    float diagonal_;         ///< diagonal length in 3d point cloud coorindate system [meters]
    cv::Matx13f cube_center_;   ///< vector to center of cube surrounding surface

    cv::Rect2i rect_;        ///< rectangle for representing 2d bounding box in pixel coordinate space
};

} // namespace ni

#endif // _NI_CORE_SURFACE_H_
