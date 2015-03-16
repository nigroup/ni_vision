#ifndef _NI_CORE_COLOR_UTILS_H_
#define _NI_CORE_COLOR_UTILS_H_

#include "elm/core/cv/typedefs_fwd.h"

namespace ni {

void normalizeColors(const cv::Mat &src, cv::Mat &dst);

}

#endif // _NI_CORE_COLOR_UTILS_H_
