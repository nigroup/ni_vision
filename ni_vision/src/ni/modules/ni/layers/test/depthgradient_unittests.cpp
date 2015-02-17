#include "ni/layers/depthgradient.h"

#include "gtest/gtest.h"

#include <memory>

#include "elm/core/exception.h"
#include "elm/core/layerconfig.h"
#include "elm/core/signal.h"
#include "elm/ts/layer_assertions.h"

using namespace std;
using namespace cv;
using namespace elm;
using namespace ni;

namespace {

ELM_INSTANTIATE_LAYER_TYPED_TEST_CASE_P(DepthGradient);

class DepthGradientTest : public ::testing::Test
{
public:

protected:
    virtual void SetUp()
    {
        to_.reset(new DepthGradient());

        config_ = LayerConfig();

        // params
        PTree params;
        config_.Params(params);

        // IO

        to_.reset(new DepthGradient(config_));
    }

    unique_ptr<base_Layer> to_; ///< test object
    LayerConfig config_;        ///< default config for tests
};

} // annonymous namespace for test cases and fixtures
