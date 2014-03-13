/*
 * Header functions
 */

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointXYZRGBNormal PointNormalT;

int min(int a, int b) {if (a > b) return b; else return a;}
float min(float a, float b) {if (a > b) return b; else return a;}
double min(double a, double b) {if (a > b) return b; else return a;}
int max(int a, int b) {if (a > b) return a; else return b;}
float max(float a, float b) {if (a > b) return a; else return b;}
double max(double a, double b) {if (a > b) return a; else return b;}

//////////////////////////////////////////////////////////////////////////////////////////////////
// pulls r,b,g fields out of packed "rgb" float in an image
void unpack_rgb(float rgb, uint8_t& r, uint8_t& g, uint8_t& b) {
        uint32_t rgbval;
        memcpy(&rgbval, &rgb, sizeof(float));

        //uint8_t garbage_ = (uint8_t)((rgb_val_ >> 24) & 0x000000ff);
        r = (uint8_t)((rgbval >> 16) & 0x000000ff);
        g = (uint8_t)((rgbval >> 8) & 0x000000ff);
        b = (uint8_t)((rgbval) & 0x000000ff);
}

// Copy parameters from one point cloud to another one.  Clouds can be of different types
template <typename PointT1, typename PointT2>
void match_cloud_params(pcl::PointCloud<PointT1>& dest, pcl::PointCloud<PointT2>& src) {
        dest.header = src.header;
        dest.width = src.width;
        dest.height = src.height;
        dest.is_dense = src.is_dense;
        dest.points.resize (dest.width * dest.height);
}

