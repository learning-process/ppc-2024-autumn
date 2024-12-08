#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_gaus3x3_mpi {

std::vector<std::pair<int, int>> generateIndicesProcessedElements(int rows, int cols);

void calculateIndicesSizesDispls(int total_elements, int cols, int num_proc, std::vector<int>& sizes,
                                 std::vector<int>& displs);

std::vector<std::vector<std::pair<int, int>>> makeWorkersIndices(const std::vector<std::pair<int, int>>& indices,
                                                                 const std::vector<int>& sizes,
                                                                 const std::vector<int>& displs);

void calculateMatrixSizesDispls(const std::vector<std::vector<std::pair<int, int>>>& matrix, int cols,
                                std::vector<int>& sizes, std::vector<int>& displs);

inline int applyGaussianKernel(const std::vector<int>& neighborhood) {
  double result = 0.0;
  result += neighborhood[0] * 0.0625;
  result += neighborhood[1] * 0.125;
  result += neighborhood[2] * 0.0625;
  result += neighborhood[3] * 0.125;
  result += neighborhood[4] * 0.25;
  result += neighborhood[5] * 0.125;
  result += neighborhood[6] * 0.0625;
  result += neighborhood[7] * 0.125;
  result += neighborhood[8] * 0.0625;
  int finalResult = static_cast<int>(std::round(result));
  return std::clamp(finalResult, 0, 255);
}

class Gaus3x3ParallelMPI : public ppc::core::Task {
 public:
  explicit Gaus3x3ParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix;
  int rows;
  int cols;
  std::vector<std::pair<int, int>> indices;
  std::vector<int> indices_sizes;
  std::vector<int> indices_displs;
  std::vector<int> worker_sizes;
  std::vector<int> worker_displs;
  std::vector<int> result_vector;
  boost::mpi::communicator world;
};

class Gaus3x3SequentialMPI : public ppc::core::Task {
 public:
  explicit Gaus3x3SequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix;
  int rows;
  int cols;
  std::vector<int> result_vector;
};

}  // namespace vasilev_s_gaus3x3_mpi
