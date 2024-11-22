#include "LayerFactory.h"

using namespace std;

namespace beednn {

    static std::unordered_map<std::string, LayerCreators>* _RegisteredLayers=NULL;
//////////////////////////////////////////////////////////////////////////////
    void LayerFactory::registerLayerType(LayerCreators& loader, const std::string type) {
        if (_RegisteredLayers == NULL)_RegisteredLayers = new std::unordered_map<std::string, LayerCreators>();
        _RegisteredLayers->insert({ {type,loader} });
    }
    //////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> LayerFactory::getAvailable(){
        std::vector<std::string> keys;
        keys.reserve(_RegisteredLayers->size());
        for (const auto& pair : *_RegisteredLayers) {
            keys.push_back(pair.first);
        }
        return keys;
    }
    //////////////////////////////////////////////////////////////////////////////
    std::string LayerFactory::getUsage(std::string sType) {
        return _RegisteredLayers->operator[](sType).constructorUsage;
    }
    //////////////////////////////////////////////////////////////////////////////
    Layer* LayerFactory::construct(const std::string& sType, std::initializer_list<float> fArgs, std::string sArg) {
        return _RegisteredLayers->operator[](sType).newCreator(fArgs,sArg);
    }
    //////////////////////////////////////////////////////////////////////////////
    Layer* LayerFactory::loadLayer(std::istream& from) {
        std::string sType;
        from >> sType;
        return _RegisteredLayers->operator[](sType).fileLoader(from);
    }
//////////////////////////////////////////////////////////////////////////////
}
