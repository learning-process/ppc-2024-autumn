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

namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi {

std::vector<std::pair<int, int>> computeProcessingIndices(int rows, int cols);

void calculateDistribution(int total_elements, int cols, int num_proc, std::vector<int>& block_sizes,
                           std::vector<int>& block_offsets);

std::vector<std::vector<std::pair<int, int>>> distributeWorkload(const std::vector<std::pair<int, int>>& indices,
                                                                 const std::vector<int>& sizes,
                                                                 const std::vector<int>& offsets);

void computeBlockDistribution(const std::vector<std::vector<std::pair<int, int>>>& data, int cols,
                              std::vector<int>& sizes, std::vector<int>& offsets);

class TaskMpi : public ppc::core::Task {
 public:
  explicit TaskMpi(std::shared_ptr<ppc::core::TaskData> data_) : Task(std::move(data_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_data;
  int height;
  int width;
  std::vector<std::pair<int, int>> processing_indices;
  std::vector<int> block_sizes;
  std::vector<int> block_offsets;
  std::vector<int> data_sizes;
  std::vector<int> data_offsets;
  std::vector<int> processed_data;
  boost::mpi::communicator comm;
};

class TaskSeq : public ppc::core::Task {
 public:
  explicit TaskSeq(std::shared_ptr<ppc::core::TaskData> data_) : Task(std::move(data_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int num_rows;
  int num_cols;
  std::vector<int> input_data;
  std::vector<int> output_data;
};

}  // namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi
