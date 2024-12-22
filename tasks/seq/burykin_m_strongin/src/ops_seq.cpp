#include "seq/burykin_m_strongin/include/ops_seq.hpp"

namespace burykin_m_strongin {

bool StronginSequential::pre_processing() {
  internal_order_test();
  res = 0;
  x0 = *reinterpret_cast<double*>(taskData->inputs[0]);
  x1 = *reinterpret_cast<double*>(taskData->inputs[1]);
  eps = *reinterpret_cast<double*>(taskData->inputs[2]);
  return true;
}

bool StronginSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool StronginSequential::run() {
  internal_order_test();
  std::vector<double> x;
  std::vector<double> y;
  double lipshM = 0.0;
  double lipshm = 0.0;
  double R = 0.0;
  size_t interval = 0;

  if (x1 > x0) {
    x.push_back(x0);
    x.push_back(x1);
  } else {
    x.push_back(x1);
    x.push_back(x0);
  }

  while (true) {
    for (size_t i = 0ull; i < x.size(); i++) {
      y.push_back(f(x[i]));
    }
    for (size_t i = 0ull; i < x.size() - 1ull; i++) {
      double lipsh = std::abs((y[i + 1ull] - y[i]) / (x[i + 1ull] - x[i]));
      if (lipsh > lipshM) {
        lipshM = lipsh;
        lipshm = lipsh + lipsh;
        double tempR = lipshm * (x[i + 1] - x[i]) + pow((y[i + 1] - y[i]), 2) / (lipshm * (x[i + 1] - x[i])) -
                       2 * (y[i + 1] + y[i]);
        if (tempR > R) {
          R = tempR;
          interval = i;
        }
      }
    }
    if (x[interval + 1] - x[interval] <= eps) {
      res = y[interval + 1];
      return true;
    }
    double newX;
    newX = (x[interval + 1] - x[interval]) / 2 + x[interval] + (y[interval + 1] - y[interval]) / (2 * lipshm);
    x.push_back(newX);
    sort(x.begin(), x.end());
    y.clear();
  }
}

bool StronginSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

}  // namespace burykin_m_strongin
