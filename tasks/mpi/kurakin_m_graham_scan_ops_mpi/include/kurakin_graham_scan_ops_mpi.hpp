#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_graham_scan_mpi {

double getRandomDouble(double start, double end);

int getRandomInt(int start, int end);

bool isLeftAngle(std::vector<double>& p1, std::vector<double>& p2, std::vector<double>& p3);

int grahamScan(std::vector<std::vector<double>>& input_);

int getCountPoint(int count_point, int size, int rank);

void getRandomVectorForGrahamScan(std::vector<double>& res_x, std::vector<double>& res_y,
                                                             int count_point, int size);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int count_point{};
  std::vector<std::vector<double>> input_;
  int count_res_point;
  std::vector<double> x_res;
  std::vector<double> y_res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  int count_point{};
  std::vector<double> input_x_;
  std::vector<double> input_y_;
  int local_count_point{};
  std::vector<double> local_input_x_;
  std::vector<double> local_input_y_;
  std::vector<std::vector<double>> local_input_;
  int count_res_point{};
};

}  // namespace kurakin_m_graham_scan_mpi