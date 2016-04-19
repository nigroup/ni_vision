#include "ni/core/img_utils.h"

void ni::indexToRowCol(int idx, int width, int& x, int& y) {

    y = idx/width;
    x = idx - y*width;
    //if (x < 0) {"Error!! converted position has negative value!\n"; x = 0;}
}
