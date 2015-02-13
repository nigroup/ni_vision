#ifndef _NI_CORE_SURFACE_H_
#define _NI_CORE_SURFACE_H_

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
    int pixelCount();

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
     * @brief Default Constructor
     */
    Surface();

protected:

    // members
    VecI pixel_indices_;    ///< indicies to pixels belonging to this surface in flattened image
    int id_;                ///< surface identifier
    int last_seen_count_;   ///< count for no. of frames since last tracked

};

} // namespace ni

#endif // _NI_CORE_SURFACE_H_
