#ifndef _NI_CORE_GRAPH_VERTEXOPSEGMENTRECT_H_
#define _NI_CORE_GRAPH_VERTEXOPSEGMENTRECT_H_

#include "elm/core/graph/base_GraphVertexOp.h"

namespace ni {

class VertexOpSegmentRect : public elm::base_GraphVertexOp
{
public:
    VertexOpSegmentRect();

    cv::Mat1f mutableOp(const cv::Mat1i& img, const cv::Mat &mask);

    static cv::Mat1f calcRect(const cv::Mat1i& img, const cv::Mat &mask);
};

} // namespace ni

#endif // _NI_CORE_GRAPH_VERTEXOPSEGMENTRECT_H_
