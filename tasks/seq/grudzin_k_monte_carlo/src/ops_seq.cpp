#include "seq/grudzin_k_monte_carlo/include/ops_seq.hpp"

namespace grudzin_k_montecarlo_seq {
template <const int dimension>
bool MonteCarloSeq<dimension>::pre_processing() {
  internal_order_test();
  dim.resize(2 * dimension);
  auto* dim_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(dim_ptr, dim_ptr + 2 * dimension, dim.data());

  auto* N_ptr = reinterpret_cast<int*>(taskData->inputs[1]);
  std::copy(N_ptr, N_ptr + 1, &N);
  return true;
}

template <const int dimension>
bool MonteCarloSeq<dimension>::run() {
  internal_order_test();

  std::mt19937 rnd(12);
  std::uniform_real_distribution<> dis(0.0, 1.0);
  result = 0.0;
  for (int i = 0; i < N; ++i) {
    std::array<double, dimension> x;
    for (int j = 0; j < 2 * dimension; j += 2) {
      x[j / 2] = dim[j] + (dim[j + 1] - dim[j]) * dis(rnd);
    }
    result += f(x);
  }
  double mult = 1.0 / (static_cast<double>(N));
  for (int j = 0; j < 2 * dimension; j += 2) {
    mult *= (dim[j + 1] - dim[j]);
  }
  result *= mult;
  return true;
}

template <const int dimension>
bool MonteCarloSeq<dimension>::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

template <const int dimension>
bool MonteCarloSeq<dimension>::validation() {
  internal_order_test();
  return !taskData->outputs_count.empty() && taskData->outputs_count[0] == 1 && taskData->inputs_count.size() == 2 &&
         taskData->inputs_count[0] == dimension;
}

template class MonteCarloSeq<1>;

template class MonteCarloSeq<2>;

template class MonteCarloSeq<3>;
}  // namespace grudzin_k_montecarlo_seq