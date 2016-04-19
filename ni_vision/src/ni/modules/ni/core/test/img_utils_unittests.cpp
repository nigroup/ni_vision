#include "ni/core/img_utils.h"

#include "gtest/gtest.h"

using namespace ni;

namespace {

TEST(ImgUtil_IndexToRowCol_Test, Index)
{
    const int ROWS=10;
    const int COLS=10;

    int index = 0;

    for(int r=0; r<ROWS; r++) {

        for(int c=0; c<COLS; c++) {

            int x = -1, y = -1;
            indexToRowCol(index, COLS, x, y);

            EXPECT_EQ(r, y);
            EXPECT_EQ(c, x);

            index++;
        }
    }
}

TEST(ImgUtil_IndexToRowCol_Test, Index_single_col)
{
    const int ROWS=10;
    const int COLS=1;

    int index = 0;

    for(int r=0; r<ROWS; r++) {

        for(int c=0; c<COLS; c++) {

            int x = -1, y = -1;
            indexToRowCol(index, COLS, x, y);

            EXPECT_EQ(r, y);
            EXPECT_EQ(c, x);

            index++;
        }
    }
}

TEST(ImgUtil_IndexToRowCol_Test, Index_negative)
{
    const int ROWS=10;
    const int COLS=1;

    int index = 0;

    for(int r=0; r<ROWS; r++) {

        for(int c=0; c<COLS; c++) {

            int x = -1, y = -1;
            indexToRowCol(-index, COLS, x, y);

            EXPECT_EQ(-r, y);
            EXPECT_EQ(-c, x);

            index++;
        }
    }
}

} // annonymous namespace for unit tests
