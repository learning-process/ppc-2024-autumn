#include "seq/burykin_m_strongin/include/ops_seq.hpp"

namespace burykin_m_strongin {

double StronginOptimization::f(double x) { return x * x; }

double StronginOptimization::strongin_method(double x0, double x1, double eps) {
  std::vector<double> x = {x0, x1};
  std::vector<double> y = {f(x0), f(x1)};

  double lipshM;
  double lipshm;
  size_t interval = 0;

  while (true) {
    lipshM = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
      double L = std::abs((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
      lipshM = std::max(lipshM, L);
    }
    lipshm = 2.0 * (lipshM > 0.0 ? lipshM : 1.0);

    double R = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < x.size() - 1; ++i) {
      double tempR = lipshm * (x[i + 1] - x[i]) + std::pow((y[i + 1] - y[i]), 2) / (lipshm * (x[i + 1] - x[i])) -
                     2.0 * (y[i + 1] + y[i]);
      if (tempR > R) {
        R = tempR;
        interval = i;
      }
    }

    if ((x[interval + 1] - x[interval]) <= eps) {
      return x[interval];
    }

    double newX = 0.5 * (x[interval + 1] + x[interval]) - 0.5 * (y[interval + 1] - y[interval]) / lipshm;
    x.push_back(newX);
    std::sort(x.begin(), x.end());
    y.clear();
    for (double xi : x) {
      y.push_back(f(xi));
    }
  }
}

bool StronginOptimization::validation() {
  internal_order_test();

  return taskData->inputs_count.size() >= 3;
}

bool StronginOptimization::pre_processing() {
  internal_order_test();

  x0 = *reinterpret_cast<double*>(taskData->inputs[0]);
  x1 = *reinterpret_cast<double*>(taskData->inputs[1]);
  epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  return true;
}

bool StronginOptimization::run() {
  internal_order_test();
  result = strongin_method(x0, x1, epsilon);
  return true;
}

bool StronginOptimization::post_processing() {
  internal_order_test();

  *reinterpret_cast<double*>(taskData->outputs[0]) = result;
  return true;
}

}  // namespace burykin_m_strongin
