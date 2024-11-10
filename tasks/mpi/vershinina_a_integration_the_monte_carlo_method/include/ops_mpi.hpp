#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_integration_the_monte_carlo_method {

std::vector<double> getRandomVector();

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<double(double)> p;

 private:
  double xmin{};
  double xmax{};
  double ymin{};
  double ymax{};
  double *input_{};
  double iter_count{};
  double reference_res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<double(double)> p;
  double xmin{};
  double xmax{};
  double ymin{};
  double ymax{};
  double iter_count{};
  double local_total;
  double local_inBox;

 private:
  std::vector<double> input_;
  double global_res{};
  boost::mpi::communicator world;
};
}  // namespace vershinina_a_integration_the_monte_carlo_method