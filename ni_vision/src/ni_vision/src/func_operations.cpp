/*
 * Functions for Histogram calculation and processing of image data
 */


int min(int a, int b) {if (a > b) return b; else return a;}
float min(float a, float b) {if (a > b) return b; else return a;}
double min(double a, double b) {if (a > b) return b; else return a;}
int max(int a, int b) {if (a > b) return a; else return b;}
float max(float a, float b) {if (a > b) return a; else return b;}
double max(double a, double b) {if (a > b) return a; else return b;}


void GetPixelPos(int idx, int width, int& x, int& y) {
    y = idx/width;
    x = idx - y*width;
    //if (x < 0) {"Error!! converted position has negative value!\n"; x = 0;}
}

int GetPixelIdx(int x, int y, int width) {
    int idx = y*width + x;
    return idx;
}


void DrawDepth(std::vector<float> vnInput, bool mode, float max, float min, cv::Mat &cvm_out)
{
    if (cvm_out.channels() == 1) {
        float scale = max-min, x;
        int g;
        for(size_t i = 0; i < vnInput.size(); i++) {
            x = vnInput[i] - min;
            g = (int)(254 * x / scale) + 1;
            if (g > 255) {printf("Depth image generation error. Gray scale exceeds 255.\n"); g = 255;}
            if (g < 1)   {printf("Depth image generation error. Gray scale is nagative.\n"); g = 1;}
            cvm_out.data[i] = g;
        }
    }
    else if (cvm_out.channels() == 3) {
        float scale =  6/(max-min), x;
        int r, g, b;

        for(size_t i = 0; i < vnInput.size(); i++) {
            if (!vnInput[i]) continue;

            x = vnInput[i] - min;
            //////////// Set RGB w.r.t distance //////////////////////////////
            // R
            if (x < (max-min)*2/3) r = (int)(510 - 255 * x * scale);
            else r = (int)(x * 255/4 * scale - 255);

            // G
            if (x < (max-min)*1/3) g = (int)(255 * x * scale);
            else g = (int)(1020 - 255 * x * scale);

            // B
            b = (int)(255 * x * scale - 510);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            if(mode){
                cvm_out.data[i*3] = r;
                cvm_out.data[i*3 + 1] = g;
                cvm_out.data[i*3 + 2] = b;
            }
            else{       // default: red-close blue-far
                cvm_out.data[i*3] = b;
                cvm_out.data[i*3 + 1] = g;
                cvm_out.data[i*3 + 2] = r;
            }
        }
    }
}

void DrawDepth(std::vector<float> vnInput, std::vector<int> index, bool mode, float max, float min, cv::Mat &cvm_out)
{
    if (cvm_out.channels() == 1) {
        float scale = max-min, x;
        int g;
        for(size_t i = 0; i < index.size(); i++) {
            x = vnInput[index[i]] - min;
            g = (int)(254 * x / scale) + 1;
            if (g > 255) {printf("Depth image generation error. Gray scale exceeds 255.\n"); g = 255;}
            //if (g < 1)   {printf("Depth image generation error. Gray scale is nagative.\n"); g = 1;}
            cvm_out.data[index[i]] = g;
        }
    }
    else if (cvm_out.channels() == 3) {
        float scale =  6/(max-min), x;
        int r, g, b;

        for(size_t i = 0; i < index.size(); i++) {
            if (!vnInput[index[i]]) continue;

            x = vnInput[index[i]] - min;
            //////////// Set RGB w.r.t distance //////////////////////////////
            // R
            if (x < (max-min)*2/3) r = (int)(510 - 255 * x * scale);
            else r = (int)(x * 255/4 * scale - 255);

            // G
            if (x < (max-min)*1/3) g = (int)(255 * x * scale);
            else g = (int)(1020 - 255 * x * scale);

            // B
            b = (int)(255 * x * scale - 510);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            if(mode){
                cvm_out.data[index[i]*3] = r;
                cvm_out.data[index[i]*3 + 1] = g;
                cvm_out.data[index[i]*3 + 2] = b;
            }
            else{       // default: red-close blue-far
                cvm_out.data[index[i]*3] = b;
                cvm_out.data[index[i]*3 + 1] = g;
                cvm_out.data[index[i]*3 + 2] = r;
            }
        }
    }
}


void CalcDepth(cv::Mat input, std::vector<int> index, float max, float min, std::vector<float>& vnOut)
{
    float scale = (max-min), x;
    int g;
    for(size_t i = 0; i < index.size(); i++) {
        g = input.data[index[i]];
        if (g) {
            x = (g-1) * scale /254 + min;
            vnOut[index[i]] = x;
        }
    }
}

void CalcDepthGrad(cv::Mat input, std::vector<int> index, float max, float min, float none, int amp, std::vector<float> &vnOut)
{
    float scale =  (max-min), x;
    uint8_t g;
    for(size_t i = 0; i < index.size(); i++) {
        g = input.data[index[i]];
        if (g) {
            x = (g-1) * scale /254 + min;
            vnOut[index[i]] = x;
        }
        else vnOut[index[i]] = none;
    }
}


void DrawDepthGrad(std::vector<float> vnInput, std::vector<int> index, bool mode, float max, float min, float none, cv::Mat &cvm_out)
{
    cvm_out = cv::Scalar(0, 0, 0);
    if (cvm_out.channels() == 1) {
        float scale = max-min, x;
        int g;
        for(size_t i = 0; i < index.size(); i++) {
            x = vnInput[index[i]] - min;
            if (x > none*0.8) g = 0;
            else {
                g = (int)(254 * x / scale) + 1;
                if (g > 255) {printf("Depth image generation error. Gray scale exceeds 255. %7.4f %7.4f %3d %f\n", x, scale, g, none); g = 255;}
            }
            cvm_out.data[index[i]] = g;
        }
    }
    if (cvm_out.channels() == 3) {
        float scale =  6/(max-min), x;
        int r, g, b;

        for(size_t i = 0; i < index.size(); i++) {
            x = vnInput[index[i]] - min;
            //////////// Set RGB w.r.t distance //////////////////////////////
            // R
            if (x < (max-min)*2/3)
                r = (int)(510 - 255 * x * scale);
            else r = (int)(x * 255/4 * scale - 255);

            // G
            if (x < (max-min)*1/3)
                g = (int)(255 * x * scale);
            else g = (int)(1020 - 255 * x * scale);

            // B
            b = (int)(255 * x * scale - 510);

            if (vnInput[index[i]] >= none*0.8) {r = 0; g = 0; b = 0;}

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            if(mode){
                cvm_out.data[index[i]*3] = r;
                cvm_out.data[index[i]*3 + 1] = g;
                cvm_out.data[index[i]*3 + 2] = b;
            }
            else{       // default: red-close blue-far
                cvm_out.data[index[i]*3] = b;
                cvm_out.data[index[i]*3 + 1] = g;
                cvm_out.data[index[i]*3 + 2] = r;
            }
        }
    }
}


// This method is used to generate the object libraries both by makelib_simple and ni_vision
// It classifies every pixel into one bin in every channel. With 3 channels and 8 bins per channel that gives
// 512 possible values for each pixel. The result is saved in a vector of that length, which saves the count
// of pixel which have a certain value. Later on this vector is then normalized
// such that the sum over all entries is 1.
void Calc3DColorHistogram(cv::Mat cvm_input, std::vector<int> index, int bin_base, std::vector<float> &vnOut) {
    if (index.size()){
        int bin_r, bin_g, bin_b;
        float r, g, b;

        float bin_width;
        // max has to be slightly greater than 1, since otherwise there is a problem if R, G or B is equal to 255
        float max = 1.0001, sum;
        bin_width = (float)max/bin_base;


        uint8_t R_, G_, B_;
        for(size_t i = 0; i < index.size(); i++) {

            B_ = cvm_input.data[index[i]*3];
            G_ = cvm_input.data[index[i]*3+1];
            R_ = cvm_input.data[index[i]*3+2];

            sum = (int)R_ + (int)G_ + (int)B_;


            if (sum) {r = (float)R_/sum; g = (float)G_/sum; b = (float)B_/sum;}
            else {r = 1/3; g = 1/3; b = 1/3;}
            bin_width = (float)max/bin_base;
            bin_r = (int)(r/bin_width);
            bin_g = (int)(g/bin_width);
            bin_b = (int)(b/bin_width);

            vnOut[bin_r*bin_base*bin_base + bin_g*bin_base + bin_b]++;
        }

        ///// normalizing //////////////
        for (int i = 0; i < bin_base*bin_base*bin_base; i++) vnOut[i] = vnOut[i]/index.size();
    }
}





void Map2Objects(std::vector<int> vnInput, int obj_nr, int nDsSize, std::vector<std::vector<int> > &mnOut, std::vector<int> &cnt)
{
    for (int i = 0; i < nDsSize; i++) if (vnInput[i]) mnOut[vnInput[i]][cnt[vnInput[i]]++] = i;
    for (int i = 0; i < obj_nr; i++) mnOut[i].resize(cnt[i]);
}


void AssignColor(int idx, cv::Scalar color, cv::Mat &cvm_out)
{
    cvm_out.data[idx*3] = color[0];
    cvm_out.data[idx*3+1] = color[1];
    cvm_out.data[idx*3+2] = color[2];
}

void Map2Image(std::vector<int> vnInput, int nDsSize, std::vector<cv::Scalar> mnColorTab, cv::Mat &cvm_out)
{
    switch (cvm_out.channels()) {
    case 1: for (int i = 0; i < nDsSize; i++) cvm_out.data[i] = mnColorTab[vnInput[i]][0]; break;
    case 3: for (int i = 0; i < nDsSize; i++) AssignColor(i, mnColorTab[vnInput[i]], cvm_out); break;
    }
}

void Idx2Image(std::vector<int> vnInput, cv::Scalar color, cv::Mat &cvm_out)
{
    switch (cvm_out.channels()) {
    case 1: for (size_t i = 0; i < vnInput.size(); i++) cvm_out.data[vnInput[i]] = color[0]; break;
    case 3: for (size_t i = 0; i < vnInput.size(); i++) AssignColor(vnInput[i], color, cvm_out); break;
    }
}
