#include "ni/core/surface.h"

using namespace std;
using namespace cv;
using namespace ni;

Surface::Surface()
{
}

int Surface::pixelCount() const
{
    return pixel_count_;
}

void Surface::overwritePixelCount(int count)
{
    pixel_count_ = count;
}

void Surface::pixelIndices(const VecI &v)
{
    pixel_indices_ = v;
    pixel_count_ = static_cast<int>(pixel_indices_.size());
}

VecI Surface::pixelIndices() const
{
    return pixel_indices_;
}

int Surface::lastSeenCount() const
{
    return last_seen_count_;
}

void Surface::lastSeenCount(bool is_last_seen)
{
    if(is_last_seen) {

        last_seen_count_ = 0;
    }
    else {
        last_seen_count_++;
    }
}

void Surface::id(int new_id)
{
    id_ = new_id;
}

int Surface::id() const
{
    return id_;
}

void Surface::colorHistogram(const Mat1f &hist)
{
    this->color_hist_ = hist;
}

Mat1f Surface::colorHistogram() const
{
    return color_hist_;
}

void Surface::diagonal(float d)
{
    diagonal_ = d;
}

float Surface::diagonal() const
{
    return diagonal_;
}

void Surface::cubeCenter(const cv::Matx13f& c)
{
    cube_center_ = c;
}

cv::Matx13f Surface::cubeCenter() const
{
    return cube_center_;
}

float Surface::distance(const Surface &s) const
{
    return static_cast<float>(cv::norm(cube_center_-s.cubeCenter(), cv::NORM_L2));
}

Rect2i Surface::rect() const
{
    return rect_;
}

void Surface::rect(const cv::Rect2i &r)
{
    rect_ = r;
}

cv::Matx23f Surface::cubeVertices() const
{
    return cube_vertices_;
}

void Surface::cubeVertices(const cv::Matx23f &cube)
{
    cube_vertices_ = cube;
}
