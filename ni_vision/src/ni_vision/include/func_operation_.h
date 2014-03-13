#define PI 3.1415926536


void GetPixelPos(int idx, int width, int& x, int& y);

int GetPixelIdx(int x, int y, int width);

void DrawDepth(std::vector<float> vnInput, bool mode, float max, float min, cv::Mat &cvm_out);

void DrawDepth(std::vector<float> vnInput, std::vector<int> index, bool mode, float max, float min, cv::Mat &cvm_out);

void CalcDepth(cv::Mat input, std::vector<int> index, float max, float min, std::vector<float>& vnOut);

void CalcDepthGrad(cv::Mat input, std::vector<int> index, float max, float min, float none, int amp, std::vector<float> &vnOut);

void DrawDepthGrad(std::vector<float> vnInput, std::vector<int> index, bool mode, float max, float min, float none, cv::Mat &cvm_out);

void CalcHistogram(std::vector<float> vnInput, std::vector<int> index, float bin_base, float input_bound, std::vector<int>& vnOut, std::vector<int>& index_map, int &bin_max_global);

void CalcColorHistogram(std::vector<float> vnInput, std::vector<int> index, int bin_base, std::vector<float> &vnOut);


// This method is used by simplemakelib to generate the object libraries
void Calc3DColorHistogram_SML1(cv::Mat &cvm_input, std::vector<int> &index, int mode, int bin_base, std::vector<float> &vnOut);


// Used by simplemakelib
// This function also computes the histogram as Calc3DColorHistogram, but it increments the two nearests bins instead of just one
void Calc3DColorHistogram_SML2(cv::Mat &cvm_input, std::vector<int> &index, int mode, int bin_base, std::vector<float> &vnOut);


void DrawHistogram(std::vector<int> vnInput, int width, int height, int bin_max_global, cv::Mat &cvm_out, int& y_max);

void DrawHistogram2(std::vector<int> vnInput, int width, int height, int bin_max_global, cv::Mat &cvm_out);

void DrawHistogram(std::vector<int> vnInput, int width, int height, int bin_max_global, int r, int g, int b, cv::Mat &cvm_out, int y_max);

void DrawColorHistogram(std::vector<float> vnInput, int width, int height, int bin_max_global, int r, int g, int b, cv::Mat &cvm_out, float y_max);

void Map2Objects(std::vector<int> vnInput, int obj_nr, int nCvSize, std::vector<std::vector<int> > &mnOut, std::vector<int> &cnt);

void AssignColor(int idx, cv::Scalar color, cv::Mat &cvm_out);

void Map2Image(std::vector<int> vnInput, int nCvSize, std::vector<cv::Scalar> mnColorTab, cv::Mat &cvm_out);

void Idx2Image(std::vector<int> vnInput, cv::Scalar color, cv::Mat &cvm_out);
