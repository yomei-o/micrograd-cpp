// ---------------------------------------------------------------------------
//  main.cpp  --  demonstrates and self-verifies the C++ micrograd port.
//
//  1) Ports the two expression tests from test/test_engine.py and checks the
//     analytic gradients against finite differences (fully self-contained --
//     no PyTorch needed). It also prints the classic micrograd reference
//     numbers so you can eyeball the match.
//  2) Trains a small MLP (built from nn.h) on XOR to show end-to-end learning.
// ---------------------------------------------------------------------------
#include "engine.h"
#include "nn.h"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

using namespace micrograd;

static bool approx(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) < tol * (1.0 + std::fabs(b));
}

// ---- test 1: the sanity-check expression from test_engine.py --------------
static bool test_sanity_check() {
    Value x = -4.0;
    Value z = 2 * x + 2 + x;
    Value q = relu(z) + z * x;
    Value h = relu(z * z);
    Value y = h + q + q * x;
    y.backward();

    // plain-double replica for a finite-difference gradient
    auto fwd = [](double x) {
        double z = 2 * x + 2 + x;
        double q = (z < 0 ? 0.0 : z) + z * x;
        double h = (z * z < 0 ? 0.0 : z * z);
        return h + q + q * x;
    };
    const double eps = 1e-6;
    double num = (fwd(x.data() + eps) - fwd(x.data() - eps)) / (2 * eps);

    std::printf("test_sanity_check:\n");
    std::printf("  y.data = %.6f            (reference -20.0)\n", y.data());
    std::printf("  x.grad = %.6f  numeric %.6f  (reference 46.0)\n", x.grad(), num);
    bool ok = approx(y.data(), -20.0) && approx(x.grad(), num, 1e-4);
    std::printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---- test 2: the "more ops" expression from test_engine.py ----------------
static bool test_more_ops() {
    Value a = -4.0, b = 2.0;
    Value c = a + b;
    Value d = a * b + pow(b, 3);
    c = c + c + 1;
    c = c + 1 + c + (-a);
    d = d + d * 2 + relu(b + a);
    d = d + 3 * d + relu(b - a);
    Value e = c - d;
    Value f = pow(e, 2);
    Value g = f / 2.0;
    g = g + 10.0 / f;
    g.backward();

    auto relu = [](double v) { return v < 0 ? 0.0 : v; };
    auto fwd = [&](double a, double b) {
        double c = a + b;
        double d = a * b + std::pow(b, 3);
        c = c + c + 1;
        c = c + 1 + c + (-a);
        d = d + d * 2 + relu(b + a);
        d = d + 3 * d + relu(b - a);
        double e = c - d;
        double f = e * e;
        double g = f / 2.0;
        g = g + 10.0 / f;
        return g;
    };
    const double eps = 1e-6;
    double na = (fwd(a.data() + eps, b.data()) - fwd(a.data() - eps, b.data())) / (2 * eps);
    double nb = (fwd(a.data(), b.data() + eps) - fwd(a.data(), b.data() - eps)) / (2 * eps);

    std::printf("test_more_ops:\n");
    std::printf("  g.data = %.4f              (reference 24.7041)\n", g.data());
    std::printf("  a.grad = %.4f numeric %.4f (reference 138.8338)\n", a.grad(), na);
    std::printf("  b.grad = %.4f numeric %.4f (reference 645.5773)\n", b.grad(), nb);
    bool ok = approx(a.grad(), na, 1e-4) && approx(b.grad(), nb, 1e-4);
    std::printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---- demo: train an MLP on XOR --------------------------------------------
static void train_xor() {
    std::printf("\ntraining a [2, 8, 8, 1] MLP on XOR:\n");
    std::mt19937 rng(1);
    MLP model(2, {8, 8, 1}, rng);

    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> Y = {-1, 1, 1, -1};   // XOR, targets in {-1, +1}

    const double lr = 0.05;
    const int steps = 200;
    for (int step = 0; step <= steps; ++step) {
        // forward + mean-squared-error loss over the 4 examples
        Value loss = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<Value> xi = {Value(X[i][0]), Value(X[i][1])};
            Value pred = model(xi)[0];
            Value diff = pred - Y[i];
            loss = loss + diff * diff;
        }
        loss = loss / (double)X.size();

        model.zero_grad();
        loss.backward();
        for (auto& p : model.parameters()) p.data() -= lr * p.grad();

        if (step % 40 == 0) std::printf("  step %3d  loss %.6f\n", step, loss.data());
    }

    std::printf("  final predictions (target):\n");
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<Value> xi = {Value(X[i][0]), Value(X[i][1])};
        double pred = model(xi)[0].data();
        bool ok = (pred > 0) == (Y[i] > 0);
        correct += ok;
        std::printf("    (%g,%g) -> %+.3f  (%+g) %s\n",
                    X[i][0], X[i][1], pred, Y[i], ok ? "OK" : "X");
    }
    std::printf("  accuracy: %d/4\n", correct);
}

int main() {
    bool ok = true;
    ok &= test_sanity_check();
    ok &= test_more_ops();
    train_xor();
    std::printf("\n%s\n", ok ? "ALL GRADIENT CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
