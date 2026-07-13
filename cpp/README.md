# micrograd — C++ port

A dependency-free **standard C++17** port of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd): a tiny scalar-valued
reverse-mode automatic differentiation (autograd) engine and a small neural-net
library built on top of it.

It ports the original [`engine.py`](../micrograd/engine.py) and
[`nn.py`](../micrograd/nn.py) to C++ header files.

## Do we need Eigen? No.

micrograd is deliberately **scalar-based**: every `Value` holds a single number
and its gradient, and the graph is built one arithmetic operation at a time.
There are no matrices, so there is **no need for Eigen** or any other
linear-algebra library. This port uses only the C++ standard library, matching
micrograd's minimal, zero-dependency spirit.

- **No PyTorch, no Eigen, no BLAS, no third-party libraries.**
- Builds and runs unchanged with **MSVC (Visual Studio 2022), GCC and Clang**,
  on Windows, Linux and macOS.
- Reproduces micrograd's classic reference gradients exactly.

## Files

| File | Purpose |
|---|---|
| `engine.h` | The `Value` autograd engine (port of `engine.py`) |
| `nn.h` | `Neuron`, `Layer`, `MLP` (port of `nn.py`) |
| `main.cpp` | Self-verifying tests + an XOR training demo |
| `CMakeLists.txt` | Cross-platform build |

## How the port works

In Python a `Value` is a reference-counted object and its `_backward` is a
closure over its children. In C++:

- `Value` is a lightweight handle holding a `std::shared_ptr<Node>`. Copying a
  `Value` shares the same node, mirroring Python's reference semantics — so the
  `parameters()` returned by the net alias the real weights, and an optimiser
  can update them in place.
- Operators (`+ - * /`, `pow`, `relu`) are overloaded to build the graph, so
  expressions read just like the Python: `auto y = 2 * x + 2 + x;`
- Each node stores a `std::function<void()>` local backward step. It captures
  its children by `shared_ptr` (a DAG node owning its children is fine) and the
  output node by raw pointer (no ownership → no reference cycle → no leak).
- `backward()` builds a topological order and applies the chain rule in reverse,
  exactly like `engine.py`.

## Build

### Visual Studio 2022 (CMake)

Open Visual Studio 2022 → **Open a Local Folder** → select this `cpp` folder.
VS auto-detects `CMakeLists.txt`; pick **`x64-Release`** and **Build → Build All**.

### Visual Studio 2022 (command line)

In the **x64 Native Tools Command Prompt for VS 2022**:

```bat
cl /O2 /EHsc /std:c++17 main.cpp
```

### CMake (any compiler)

```bash
cmake -S . -B build
cmake --build build --config Release
```

### g++ / clang++ directly

```bash
g++ -O2 -std=c++17 -o micrograd main.cpp
# or
clang++ -O2 -std=c++17 -o micrograd main.cpp
```

## Run

```
micrograd
```

Expected output:

```
test_sanity_check:
  y.data = -20.000000            (reference -20.0)
  x.grad = 46.000000  numeric 46.000000  (reference 46.0)
  -> PASS
test_more_ops:
  g.data = 24.7041              (reference 24.7041)
  a.grad = 138.8338 numeric 138.8338 (reference 138.8338)
  b.grad = 645.5773 numeric 645.5773 (reference 645.5773)
  -> PASS

training a [2, 8, 8, 1] MLP on XOR:
  step   0  loss 1.063664
  ...
  step 200  loss 0.007625
  final predictions (target):
    (0,0) -> -0.989  (-1) OK
    (0,1) -> +0.878  (+1) OK
    (1,0) -> +0.956  (+1) OK
    (1,1) -> -1.111  (-1) OK
  accuracy: 4/4

ALL GRADIENT CHECKS PASSED
```

The two expression tests are the same ones as `test/test_engine.py`. Their
gradients are checked against finite differences (so the check is
self-contained, no PyTorch required), and they match the well-known micrograd
reference values that `test_engine.py` compares to PyTorch.

## Using the library

```cpp
#include "engine.h"
#include "nn.h"
using namespace micrograd;

Value a = 2.0, b = -3.0;
Value c = a * b + relu(a + b);
c.backward();
// a.grad(), b.grad() now hold d c / d a, d c / d b

std::mt19937 rng(1);
MLP model(2, {8, 8, 1}, rng);   // 2 inputs -> 8 -> 8 -> 1 (ReLU hidden, linear out)
std::vector<Value> x = {Value(0.5), Value(-1.0)};
Value y = model(x)[0];
```
