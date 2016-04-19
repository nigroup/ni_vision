#ifndef _NI_IMG_UTILS_H
#define IMG_UTILS_H

namespace ni {

/**
 * @brief Convert vector index to 2D-matrix indices (row, column)
 * @param[in] idx
 * @param[in] width
 * @param[out] x
 * @param[out] y
 */
void indexToRowCol(int idx, int width, int& x, int& y);

}

#endif // IMG_UTILS_H
