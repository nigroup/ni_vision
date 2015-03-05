#ifndef _NI_CORE_GRAPH_VERTEXOPSEGMENTSIZE_H_
#define _NI_CORE_GRAPH_VERTEXOPSEGMENTSIZE_H_

#include "elm/core/graph/base_GraphVertexOp.h"

namespace ni {

/**
 * @brief class for measuring segment size
 */
class VertexOpSegmentSize : public elm::base_GraphVertexOp
{
public:
    VertexOpSegmentSize();

    cv::Mat1f mutableOp(const cv::Mat1f& img, const cv::Mat1b &mask);

    static cv::Mat1f calcSize(const cv::Mat1f& img, const cv::Mat1b &mask);

protected:
    int size_;
};

} // namespace ni

#endif // _NI_CORE_GRAPH_VERTEXOPSEGMENTSIZE_H_
