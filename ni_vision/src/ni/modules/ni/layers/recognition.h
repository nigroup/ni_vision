#ifndef _NI_LAYERS_RECOGNITION_H_
#define _NI_LAYERS_RECOGNITION_H_

#include <vector>

#include <flann/flann.h>

#include "elm/core/typedefs_fwd.h"
#include "elm/core/pcl/typedefs_fwd.h"
#include "elm/layers/layers_interim/base_matoutputlayer.h"

#include "ni/core/surface.h"
#include "ni/legacy/surfprop.h"
#include "ni/legacy/trackprop.h"

namespace ni {

/**
 * @brief layer for comparing a selected surface
 * with the trained model of an object
 * @cite Mohr2014
 *
 * Output keys defined by parent
 */
class Recognition : public elm::base_MatOutputLayer
{
public:


protected:

};

} // namespace ni

#endif // _NI_LAYERS_RECOGNITION_H_
