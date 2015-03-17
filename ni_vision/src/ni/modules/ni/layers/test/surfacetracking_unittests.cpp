#include "ni/layers/surfacetracking.h"

#include "elm/core/debug_utils.h"
#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/inputname.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"

using namespace cv;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(SurfaceTracking);

const std::string NAME_IN_BGR       = "bgr";
const std::string NAME_IN_POINTS    = "points";
const std::string NAME_IN_MAP       = "map";

class SurfaceTrackingTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {

    }


};

TEST_F(SurfaceTrackingTest, Activate)
{

}

} // annonymous namespace for testing
