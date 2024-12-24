#include "seq/vershinina_a_cannons_algorithm/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool vershinina_a_cannons_algorithm::TestTaskSequential::pre_processing() {
  internal_order_test();
  n = taskData->inputs_count[0];

  lhs_.n = n;
  rhs_.n = n;

  res_c.n = n;
  res_.n = n;

  lhs_.read(reinterpret_cast<double*>(taskData->inputs[0]));
  rhs_.read(reinterpret_cast<double*>(taskData->inputs[1]));
  res_c = TMatrix<double>::create(n);
  res_ = TMatrix<double>::create(n);

  lhs_.hshift = 0;
  rhs_.vshift = 0;

  return true;
}

bool vershinina_a_cannons_algorithm::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 2 && taskData->inputs_count[0] > 0;
}

bool vershinina_a_cannons_algorithm::TestTaskSequential::run() {
  internal_order_test();
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res_c.at(i, j) = lhs_.at_h(i, j) * rhs_.at_v(i, j);
      }
    }
    for (int t = 0; t < n; t++) {
      for (int s = 0; s < n; s++) {
        res_.at(t, s) += res_c.at(t, s);
      }
    }
    lhs_.hshift++;
    rhs_.vshift++;
  }
  return true;
}

bool vershinina_a_cannons_algorithm::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(res_.data.begin(), res_.data.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}
