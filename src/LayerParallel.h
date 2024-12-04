#pragma once
#include <vector>
#include "Matrix.h"
#include "Layer.h"
#include "ParallelReduction.h"

namespace beednn {
	class LayerParallel : public Layer
	{
	public:
		explicit LayerParallel(std::vector<Layer*> mParallelLayers, ParallelReduction mReduction);//todo: how to connect many outputs
		virtual ~LayerParallel() override;

		virtual Layer* clone() const override;

		virtual bool init(size_t& in, size_t& out, bool debug = false) override;

		virtual bool has_weights() const override;
		virtual std::vector<MatrixFloat*> weights() const override;
		virtual std::vector<MatrixFloat*> gradient_weights() const override;

		virtual void save(std::ostream& to)const override;
		static Layer* load(std::istream& from);
		static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
		static std::string constructUsage();

		virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
		virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

	private:
		LayerParallel();
		std::vector < Layer*> _Layers;
		ParallelReduction _ParallelReduction;
	protected:
        virtual std::string generate_mermaid_node() const override {
            std::stringstream ss;
            ss << "    subgraph " << node_id << "[" << get_layer_name() << "]\n";
            if (_ParallelReduction == SUM)
                ss << "    direction LR\n"; // Left to right for parallel paths

            // Add nodes for each parallel layer
            for (const auto* layer : _Layers) {
                ss << layer->generate_mermaid_node() << "\n";
            }

            // Add reduction node
            std::string reduction_name;
            switch (_ParallelReduction) {
            case SUM: reduction_name = "Sum"; break;
            case DOT: reduction_name = "Dot"; break;
            case ROWSTACK: reduction_name = "RowStack"; break;
            case COLSTACK: reduction_name = "ColStack"; break;
            default: reduction_name = "Unknown"; break;
            }
            ss << "    " << node_id << "_reduce[\"" << reduction_name << "\"]\n";
            ss << "    end\n";
            return ss.str();
        }

        virtual std::string generate_mermaid_connections() const override {
            std::stringstream ss;

            // Connect inputs to parallel layers
            for (const auto* input : input_connections) {
                for (const auto* layer : _Layers) {
                    ss << "    " << input->node_id << " --> " << layer->node_id << "\n";
                }
            }

            // Connect parallel layers to reduction
            for (const auto* layer : _Layers) {
                ss << "    " << layer->node_id << " --> " << node_id << "_reduce\n";
            }

            return ss.str();
        }
	};
	REGISTER_LAYER(LayerParallel, "LayerParallel");
}