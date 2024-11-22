#pragma once

#include <string>
#include <istream>
#include <functional>
#include <unordered_map>

#define REGISTER_LAYER(LayerType, typeName) \
    namespace { \
        struct LayerType##Registrar { \
            LayerType##Registrar() { \
                LayerCreators creators; \
                creators.fileLoader = [](std::istream& from) -> Layer* { \
                    return LayerType::load(from); \
                }; \
                creators.newCreator = [](std::initializer_list<float>& fArgs, std::string& initializer) -> Layer* { \
                    return LayerType::construct(fArgs, initializer); \
                }; \
                creators.constructorUsage = LayerType::constructUsage(); \
                LayerFactory::registerLayerType(creators, typeName); \
            } \
        }; \
        static LayerType##Registrar LayerType##registrar; \
    }

namespace beednn {
class Layer;
struct LayerCreators{
    std::function<Layer* (std::istream&)> fileLoader;
    std::function<Layer* (std::initializer_list<float>&,std::string&)> newCreator;
    std::string constructorUsage;
};
class LayerFactory
{
public:
    static void registerLayerType(LayerCreators& loader,const std::string type);

    static std::vector<std::string> getAvailable();
    static std::string getUsage(std::string sType);
    static Layer* construct(const std::string& sType, std::initializer_list<float> fArgs,std::string sArg);
    static Layer* loadLayer(std::istream& from);
};
}
