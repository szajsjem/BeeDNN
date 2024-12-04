#include "ParallelReduction.h"

namespace beednn {
    ParallelReduction reductionFromString(const std::string& reductionStr) {
        // Convert to uppercase for case-insensitive comparison
        std::string upperStr = reductionStr;
        std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);

        if (upperStr == "ROWSTACK") return ROWSTACK;
        if (upperStr == "COLSTACK") return COLSTACK;
        if (upperStr == "SUM") return SUM;
        if (upperStr == "DOT") return DOT;

        throw std::invalid_argument("Invalid reduction type: " + reductionStr);
    }
    std::string reductionToString(ParallelReduction reduction) {
        switch (reduction) {
        case ROWSTACK: return "ROWSTACK";
        case COLSTACK: return "COLSTACK";
        case SUM: return "SUM";
        case DOT: return "DOT";
        default: return "UNKNOWN";
        }
    }
    std::vector<std::string> getAllReductionNames() {
        return { "ROWSTACK", "COLSTACK", "SUM", "DOT" };
    }
}