// mpi   rectangle method
#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

#define func double(const std::vector<double> &)

namespace rezantseva_a_rectangle_method_mpi {

bool check_integration_bounds(std::vector<std::pair<double, double>> *ib);

template <class Archive, typename T1, typename T2>
void serialize(Archive &ar, std::pair<T1, T2> &p, const unsigned int version) {
  ar & p.first;
  ar & p.second;
}

class RectangleMethodSequential : public ppc::core::Task {
 public:
  explicit RectangleMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<func> f)
      : Task(std::move(taskData_)), func_(std::move(f)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double result_{};
  std::vector<std::pair<double, double>> integration_bounds_;
  std::vector<int> distribution_;
  std::function<func> func_;
};

class RectangleMethodMPI : public ppc::core::Task {
 public:
  explicit RectangleMethodMPI(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<func> f)
      : Task(std::move(taskData_)), func_(std::move(f)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  double result_{};
  int num_processes_ = 0;
  std::vector<std::pair<double, double>> integration_bounds_;
  std::vector<int> distribution_;
  std::function<func> func_;
  int n_;
  std::vector<double> widths_;
  int total_points_{};
};
}  // namespace rezantseva_a_rectangle_method_mpi
