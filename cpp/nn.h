// ---------------------------------------------------------------------------
//  nn.h  --  C++ port of micrograd's nn.py: Neuron, Layer, MLP.
//
//  Built entirely on top of the scalar Value engine in engine.h. Parameters
//  are returned as Value handles that share the underlying nodes, so an
//  optimiser can read .grad() and write .data() in place after backward().
// ---------------------------------------------------------------------------
#ifndef MICROGRAD_NN_H
#define MICROGRAD_NN_H

#include "engine.h"
#include <random>
#include <vector>

namespace micrograd {

// A single neuron: dot(w, x) + b, optionally passed through ReLU.
struct Neuron {
    std::vector<Value> w;
    Value b{0.0};
    bool nonlin;

    Neuron(int nin, bool nonlin_, std::mt19937& rng) : nonlin(nonlin_) {
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (int i = 0; i < nin; ++i) w.emplace_back(u(rng));
    }

    Value operator()(const std::vector<Value>& x) const {
        Value act = b;
        for (size_t i = 0; i < w.size(); ++i) act = act + w[i] * x[i];
        return nonlin ? relu(act) : act;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p = w;
        p.push_back(b);
        return p;
    }
};

// A layer of independent neurons.
struct Layer {
    std::vector<Neuron> neurons;

    Layer(int nin, int nout, bool nonlin, std::mt19937& rng) {
        for (int i = 0; i < nout; ++i) neurons.emplace_back(nin, nonlin, rng);
    }

    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        out.reserve(neurons.size());
        for (const auto& nrn : neurons) out.push_back(nrn(x));
        return out;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (const auto& nrn : neurons) {
            auto np = nrn.parameters();
            p.insert(p.end(), np.begin(), np.end());
        }
        return p;
    }
};

// A multi-layer perceptron. Hidden layers use ReLU, the final layer is linear.
struct MLP {
    std::vector<Layer> layers;

    // nin inputs, then one entry in `nouts` per layer (last is the output size)
    MLP(int nin, const std::vector<int>& nouts, std::mt19937& rng) {
        std::vector<int> sz;
        sz.push_back(nin);
        sz.insert(sz.end(), nouts.begin(), nouts.end());
        for (size_t i = 0; i < nouts.size(); ++i) {
            bool nonlin = (i != nouts.size() - 1);
            layers.emplace_back(sz[i], sz[i + 1], nonlin, rng);
        }
    }

    std::vector<Value> operator()(std::vector<Value> x) const {
        for (const auto& layer : layers) x = layer(x);
        return x;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (const auto& layer : layers) {
            auto lp = layer.parameters();
            p.insert(p.end(), lp.begin(), lp.end());
        }
        return p;
    }

    void zero_grad() {
        for (auto& p : parameters()) p.grad() = 0.0;
    }
};

} // namespace micrograd

#endif // MICROGRAD_NN_H
