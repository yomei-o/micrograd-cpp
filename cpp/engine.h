// ---------------------------------------------------------------------------
//  engine.h  --  a dependency-free C++ port of Andrej Karpathy's micrograd
//                scalar-valued reverse-mode autograd engine (engine.py).
//
//  micrograd is deliberately *scalar* based: every Value holds a single number
//  and its gradient. There are no matrices, so there is no need for Eigen or
//  any other linear-algebra library -- this port uses only the C++ standard
//  library, exactly matching micrograd's "minimal, zero-dependency" spirit.
//
//  A Value is a lightweight handle (shared_ptr) to a Node in the computation
//  graph, so copying a Value shares the same underlying node -- mirroring
//  Python's reference semantics. Operators build the graph; backward() runs
//  reverse-mode automatic differentiation over it.
// ---------------------------------------------------------------------------
#ifndef MICROGRAD_ENGINE_H
#define MICROGRAD_ENGINE_H

#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <unordered_set>

namespace micrograd {

struct Node {
    double data;
    double grad = 0.0;
    std::vector<std::shared_ptr<Node>> prev;   // children this node was built from
    std::function<void()> _backward = [] {};   // local chain-rule step
    std::string op;                            // for debugging / graphviz
    explicit Node(double d) : data(d) {}
};

using NodePtr = std::shared_ptr<Node>;

class Value {
public:
    NodePtr n;

    Value() : n(std::make_shared<Node>(0.0)) {}
    Value(double d) : n(std::make_shared<Node>(d)) {}   // implicit: allows Value x = 3.0;
    explicit Value(NodePtr p) : n(std::move(p)) {}

    double  data() const { return n->data; }
    double& data()       { return n->data; }
    double  grad() const { return n->grad; }
    double& grad()       { return n->grad; }
    const std::string& op() const { return n->op; }

    // Reverse-mode autodiff: topologically sort the graph, seed this node's
    // gradient with 1, then apply each node's local backward in reverse order.
    void backward() {
        std::vector<Node*> topo;
        std::unordered_set<Node*> visited;
        std::function<void(Node*)> build = [&](Node* v) {
            if (visited.insert(v).second) {
                for (auto& c : v->prev) build(c.get());
                topo.push_back(v);
            }
        };
        build(n.get());

        n->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward();
    }
};

// -- helper to allocate an output node wired to its children ----------------
inline Value make_op(double data, std::vector<NodePtr> children, const char* op) {
    auto out = std::make_shared<Node>(data);
    out->op = op;
    out->prev = std::move(children);
    return Value(out);
}

// The backward closures capture children by shared_ptr (the graph is a DAG, so
// a node owning its children introduces no reference cycle) and the output
// node by raw pointer (no ownership -> no self-cycle -> no leak).

inline Value operator+(const Value& a, const Value& b) {
    Value out = make_op(a.data() + b.data(), {a.n, b.n}, "+");
    Node* o = out.n.get(); NodePtr an = a.n, bn = b.n;
    out.n->_backward = [o, an, bn] {
        an->grad += o->grad;
        bn->grad += o->grad;
    };
    return out;
}

inline Value operator*(const Value& a, const Value& b) {
    Value out = make_op(a.data() * b.data(), {a.n, b.n}, "*");
    Node* o = out.n.get(); NodePtr an = a.n, bn = b.n;
    out.n->_backward = [o, an, bn] {
        an->grad += bn->data * o->grad;
        bn->grad += an->data * o->grad;
    };
    return out;
}

// only integer/float powers, like micrograd
inline Value pow(const Value& a, double p) {
    Value out = make_op(std::pow(a.data(), p), {a.n}, "**");
    Node* o = out.n.get(); NodePtr an = a.n;
    out.n->_backward = [o, an, p] {
        an->grad += (p * std::pow(an->data, p - 1.0)) * o->grad;
    };
    return out;
}

inline Value relu(const Value& a) {
    double d = a.data() < 0.0 ? 0.0 : a.data();
    Value out = make_op(d, {a.n}, "ReLU");
    Node* o = out.n.get(); NodePtr an = a.n;
    out.n->_backward = [o, an] {
        an->grad += (o->data > 0.0 ? 1.0 : 0.0) * o->grad;
    };
    return out;
}

// -- derived ops (match engine.py's __neg__, __sub__, __truediv__, r*) ------
inline Value operator-(const Value& a)                 { return a * Value(-1.0); }
inline Value operator-(const Value& a, const Value& b) { return a + (-b); }
inline Value operator/(const Value& a, const Value& b) { return a * pow(b, -1.0); }

// mixed Value/double overloads so `2 * x + 2` etc. read like the Python
inline Value operator+(const Value& a, double b) { return a + Value(b); }
inline Value operator+(double a, const Value& b) { return Value(a) + b; }
inline Value operator*(const Value& a, double b) { return a * Value(b); }
inline Value operator*(double a, const Value& b) { return Value(a) * b; }
inline Value operator-(const Value& a, double b) { return a - Value(b); }
inline Value operator-(double a, const Value& b) { return Value(a) - b; }
inline Value operator/(const Value& a, double b) { return a / Value(b); }
inline Value operator/(double a, const Value& b) { return Value(a) / b; }

} // namespace micrograd

#endif // MICROGRAD_ENGINE_H
