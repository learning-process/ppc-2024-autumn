// Copyright 2024 Nesterov Alexander
#include "seq/rezantseva_a_simple_iteration_method/include/ops_seq_rezantseva.hpp"

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::isTimeToStop(
    const std::vector<double>& x0, const std::vector<double>& x1) const {
  double max_precision = 0.0;  // max precision between iterations

  for (size_t k = 0; k < x0.size(); k++) {
    double precision = std::fabs(x1[k] - x0[k]);  // |x1^(i+1) - x1^i|
    if (precision > max_precision) {
      max_precision = precision;
    }
  }
  return (max_precision < epsilon_);
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::checkMatrix() {
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

  for (size_t i = 0; i < n; ++i) {  // row

    double Aii = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + i]);
    double sum = 0.0;

    for (size_t j = 0; j < n; ++j) {  // column
      if (i != j) {
        sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + j]);
      }
    }
    if (Aii <= sum) {
      /* std::cerr << "Error: The convergence condition is not met " << std::endl;*/
      return false;
    }
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::validation() {
  internal_order_test();
  // на вход размер n, коэф. матрицы ј, вектора b == 3
  // на выход вектор приближенных решений системы == 1
  // размер матрицы больше нул€
  // провер€ем условие сходимости
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  /*
  if ((taskData->inputs_count.size() != 3) || (taskData->outputs_count.size() != 1)) {
     std::cerr << "wrong size" << std::endl;
     return false;
   }
   if (n <= 0) {
     std::cerr << "matrix size must be positive" << std::endl;
     return false;
   }
   if (!checkMatrix()) {
     std::cerr << "wrong matrix" << std::endl;
     return false;
   }
   return true;
   */
  return (taskData->inputs_count.size() == 3) && (taskData->outputs_count.size() == 1) && (n > 0) &&
         (checkMatrix() == true);
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::pre_processing() {
  internal_order_test();
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  A_.resize(n * n);
  b_.resize(n);
  x_.resize(n, 0.0);
  // fill matrix A and vector b
  for (size_t i = 0; i < n; ++i) {    // row
    for (size_t j = 0; j < n; ++j) {  // column
      A_[i * n + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n + j];
    }
    b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
  }

  return true;
  return true;
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::run() {
  internal_order_test();
  size_t iteration = 0;
  size_t n = b_.size();
  std::vector<double> x0(n, 0.0);

  while (iteration < maxIteration_) {
    std::copy(x_.begin(), x_.end(), x0.begin());  // move previous decisions to vec x0
    for (size_t i = 0; i < n; i++) {
      double sum = 0;
      for (size_t j = 0; j < n; j++) {
        if (j != i) {
          sum += A_[i * n + j] * x0[j];  // example: A12*x2 + A13*x3+..+ A1n*xn
        }
      }
      x_[i] = (b_[i] - sum) / A_[i * n + i];
    }
    if (isTimeToStop(x0, x_)) {
      break;
    }
    iteration++;
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < x_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
  }
  return true;
}
