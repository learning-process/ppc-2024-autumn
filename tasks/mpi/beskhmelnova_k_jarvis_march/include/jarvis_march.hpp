#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace beskhmelnova_k_jarvis_march_mpi {

template <typename DataType>
DataType crossProduct(const std::vector<DataType>& p1, const std::vector<DataType>& p2,
                      const std::vector<DataType>& p3);

template <typename DataType>
bool isLeftAngle(const std::vector<DataType>& p1, const std::vector<DataType>& p2, const std::vector<DataType>& p3);

template <typename DataType>
void jarvisMarch(int& num_points, std::vector<std::vector<DataType>>& input, std::vector<DataType>& res_x,
                 std::vector<DataType>& res_y);

int localNumPoints(int num_points, int world_size, int rank);

template <typename DataType>
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int num_points{};
  std::vector<std::vector<DataType>> input;
  std::vector<DataType> res_x;
  std::vector<DataType> res_y;
};

template <typename DataType>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;

  int num_points{};
  std::vector<DataType> input_x;
  std::vector<DataType> input_y;
  std::vector<DataType> res_x;
  std::vector<DataType> res_y;

  int local_num_points{};
  std::vector<DataType> local_input_x;
  std::vector<DataType> local_input_y;
  std::vector<std::vector<DataType>> local_input;
};
}  // namespace beskhmelnova_k_jarvis_march_mpi