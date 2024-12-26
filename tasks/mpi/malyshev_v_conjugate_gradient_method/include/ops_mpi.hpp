#pragma once
#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_conjugate_gradient_method_mpi {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::vector<std::vector<double>> A,
                              std::vector<double> b)
      : Task(std::move(taskData_)), A_(std::move(A)), b_(std::move(b)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 protected:
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> res_;

  virtual void optimize();

  void readTaskData();
  void writeTaskData();
  bool validateTaskData();

  std::vector<double> matrixVectorProduct(const std::vector<std::vector<double>> &A, const std::vector<double> &v);
  double dotProduct(const std::vector<double> &a, const std::vector<double> &b);
  std::vector<double> vectorAdd(const std::vector<double> &a, const std::vector<double> &b);
  std::vector<double> vectorSubtract(const std::vector<double> &a, const std::vector<double> &b);
  std::vector<double> vectorScale(const std::vector<double> &v, double scalar);
};

class TestTaskParallel : public TestTaskSequential {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::vector<std::vector<double>> A,
                            std::vector<double> b)
      : TestTaskSequential(std::move(taskData_), std::move(A), std::move(b)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
};

}  // namespace malyshev_v_conjugate_gradient_method_mpi