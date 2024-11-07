#pragma once
#include <gtest/gtest.h>
#include <functional>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_integration_the_monte_carlo_method {

std::vector<float> getRandomVector(float sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<float(float)> p;

 private:
  float xmin{};
  float xmax{};
  float ymin{};
  float ymax{};
  float *input_{};
  float reference_res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<float(float)> p;
  float xmin{};
  float xmax{};
  float ymin{};
  float ymax{};
  float local_total;
  float local_inBox;

 private:
  std::vector<float> input_;
  float global_res{};
  boost::mpi::communicator world;
};
}  // namespace vershinina_a_integration_the_monte_carlo_method