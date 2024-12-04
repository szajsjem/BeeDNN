#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace beednn {
	enum ParallelReduction {
		ROWSTACK,
		COLSTACK,
		SUM,
		DOT,
	};
    ParallelReduction reductionFromString(const std::string& reductionStr);
	std::string reductionToString(ParallelReduction reduction);
	std::vector<std::string> getAllReductionNames();
}