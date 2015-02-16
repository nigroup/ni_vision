#ifndef _NI_LAYERS_LAYERFACTORYNI_H_
#define _NI_LAYERS_LAYERFACTORYNI_H_

#include <memory>
#include <string>

#include "elm/core/base_Layer.h"
#include "elm/core/exception.h"

namespace ni {

/**
 * @brief class for implementing layer related factory methods
 * such as instantiation and sequencing of multiple layer applications (e.g. pipeline)
 */
class LayerFactoryNI
{
public:
    typedef std::string LayerType;
    typedef std::shared_ptr<elm::base_Layer> LayerShared; ///< convinience typedef to shared pointer to layer object

    LayerFactoryNI();

    /**
     * @brief Create smart pointer to an instantiated layer
     * @param type
     * @return pointer to layer instance
     * @throws ExceptionTypeError on unrecognized layer type
     */
    static LayerShared CreateShared(const LayerType &type);

    /**
     * @brief Create smart pointer to an instantiated layer
     * @param type
     * @param configuration
     * @param I/O names
     * @return pointer to layer instance
     * @throws ExceptionTypeError on unrecognized layer type
     */
    static LayerShared CreateShared(const LayerType &type,
                                    const elm::LayerConfig &config,
                                    const elm::LayerIONames &io);
};

} // namesapce ni

#endif // _NI_LAYERS_LAYERFACTORYNI_H_
