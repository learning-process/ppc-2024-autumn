#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <thread>
#include <vector>

std::pair<std::vector<int>, std::vector<int>> calculateDistribution(int cols, int rows, int num_processes) {
  std::vector<int> distribution(num_processes, 0);
  std::vector<int> displacement(num_processes, 0);

  int cols_per_proc = cols / num_processes;
  int remainder = cols % num_processes;

  int offset = 0;
  for (int i = 0; i < num_processes; ++i) {
    distribution[i] = (cols_per_proc + (i < remainder)) * rows;
    displacement[i] = offset;
    offset += distribution[i];
  }
  return std::make_pair(distribution, displacement);
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::validation() {
  internal_order_test();
  if (!taskData) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr) {
    return false;
  }
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* input_matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
    int* input_vector_ = reinterpret_cast<int*>(taskData->inputs[1]);

    int count = taskData->inputs_count[0];
    cols_ = taskData->inputs_count[1];
    rows_ = count / cols_;

    // Copy data from TaskData, handle potential errors
    try {
      input_matrix_1.assign(input_matrix_, input_matrix_ + count);
      input_vector_1.assign(input_vector_, input_vector_ + cols_);
      result_vector_.resize(rows_, 0);
    } catch (const std::exception& e) {
      std::cerr << "Error copying data: " << e.what() << std::endl;
      return false;
    }

    // Calculate distribution and displacement
    auto [distribution_result, displacement_result] = calculateDistribution(cols_, rows_, world.size());
    distribution = distribution_result;
    displacement = displacement_result;

  } else {
    // Other processes resize their local vectors to the correct size
    input_matrix_1.resize(0);
    input_vector_1.resize(0);
    result_vector_.resize(0);
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, rows_, 0);           // Broadcast number of rows
  boost::mpi::broadcast(world, input_vector_1, 0);  // Broadcast input vector
  boost::mpi::broadcast(world, distribution, 0);    // Broadcast column distribution

  int local_cols = distribution[world.rank()] / rows_;  // Number of columns per process
  int local_elements = distribution[world.rank()];      // Number of elements per process
  std::vector<int> local_matrix(local_elements);        // Local Matrix to receive data

  // Distribute matrix columns using scatterv
  std::vector<int> sendcounts = distribution;     // Send counts for scatterv
  std::vector<int> displacements = displacement;  // Displacements for scatterv

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, input_matrix_1.data(), sendcounts, displacements, local_matrix.data(), local_elements, 
                         0);
  } else {
    boost::mpi::scatterv(world, local_matrix.data(), local_elements, 0);
  }

  std::vector<int> local_result(rows_, 0);  // Initialize local result vector

  // Local matrix-vector multiplication (vertical ribbon)
  for (int j = 0; j < local_cols; ++j) {
    for (int i = 0; i < rows_; ++i) {
      local_result[i] += local_matrix[j * rows_ + i] * input_vector_1[displacement[world.rank()] / rows_ + j];
    }
  }

  // Gather results (similar to horizontal scheme's gatherv)
  std::vector<int> gather_counts;
  std::vector<int> gather_displacements;

  if (world.rank() == 0) {
    gather_counts.resize(world.size());
    gather_displacements.resize(world.size());
    gather_displacements[0] = 0;
    for (int i = 0; i < world.size(); ++i) {
      gather_counts[i] = rows_;
      if (i > 0) gather_displacements[i] = gather_displacements[i - 1] + gather_counts[i - 1];
    }
  }

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), result_vector_.data(), gather_counts,
                        gather_displacements, 0);
  } else {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), 0);
  }

  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* answer = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_vector_.begin(), result_vector_.end(), answer);
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::validation() {
  internal_order_test();
  if (!taskData) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr) {
    return false;
  }
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::pre_processing() {
  internal_order_test();

  matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
  count = taskData->inputs_count[0];
  vector_ = reinterpret_cast<int*>(taskData->inputs[1]);
  cols_ = taskData->inputs_count[1];
  rows_ = count / cols_;
  input_vector_.assign(vector_, vector_ + cols_);
  result_vector_.assign(rows_, 0);

  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::run() {
  internal_order_test();

  for (int j = 0; j < cols_; ++j) {
    for (int i = 0; i < rows_; ++i) {
      result_vector_[i] += matrix_[i + j * rows_] * input_vector_[j];
    }
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}