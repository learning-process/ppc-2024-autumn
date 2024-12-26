#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();

  readTaskData();

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::validation() {
  internal_order_test();

  return validateTaskData();
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::run() {
  internal_order_test();

  optimize();

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::post_processing() {
  internal_order_test();

  writeTaskData();

  return true;
}

void malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::readTaskData() {
  A_ = *reinterpret_cast<std::vector<std::vector<double>> *>(taskData->inputs[0]);
  b_ = *reinterpret_cast<std::vector<double> *>(taskData->inputs[1]);
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::validateTaskData() {
  return taskData != nullptr && taskData->inputs.size() >= 2 && !taskData->outputs.empty() &&
         taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr && taskData->outputs[0] != nullptr;
}

void malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::writeTaskData() {
  *reinterpret_cast<std::vector<double> *>(taskData->outputs[0]) = res_;
}

void malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::optimize() {
  size_t n = b_.size();
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;
  double rsold = dotProduct(r, r);

  for (size_t i = 0; i < n; ++i) {
    std::vector<double> Ap = matrixVectorProduct(A_, p);
    double alpha = rsold / dotProduct(p, Ap);
    x = vectorAdd(x, vectorScale(p, alpha));
    r = vectorSubtract(r, vectorScale(Ap, alpha));
    double rsnew = dotProduct(r, r);
    if (sqrt(rsnew) < 1e-10) break;
    p = vectorAdd(r, vectorScale(p, rsnew / rsold));
    rsold = rsnew;
  }

  res_ = x;
}

std::vector<double> malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::matrixVectorProduct(
    const std::vector<std::vector<double>> &A, const std::vector<double> &v) {
  std::vector<double> result(A.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i) {
    for (size_t j = 0; j < v.size(); ++j) {
      result[i] += A[i][j] * v[j];
    }
  }
  return result;
}

double malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::dotProduct(const std::vector<double> &a,
                                                                                const std::vector<double> &b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

std::vector<double> malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::vectorAdd(
    const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::vectorSubtract(
    const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::vectorScale(
    const std::vector<double> &v, double scalar) {
  std::vector<double> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = v[i] * scalar;
  }
  return result;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) readTaskData();
  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) || validateTaskData();
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::run() {
  internal_order_test();

  broadcast(world, A_, 0);
  broadcast(world, b_, 0);
  optimize();

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) writeTaskData();
  return true;
}

void malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::optimize() {
  size_t n = b_.size();
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;
  double rsold = dotProduct(r, r);

  for (size_t i = 0; i < n; ++i) {
    std::vector<double> Ap = matrixVectorProduct(A_, p);
    double alpha = rsold / dotProduct(p, Ap);
    x = vectorAdd(x, vectorScale(p, alpha));
    r = vectorSubtract(r, vectorScale(Ap, alpha));
    double rsnew = dotProduct(r, r);
    if (sqrt(rsnew) < 1e-10) break;
    p = vectorAdd(r, vectorScale(p, rsnew / rsold));
    rsold = rsnew;
  }

  res_ = x;
}