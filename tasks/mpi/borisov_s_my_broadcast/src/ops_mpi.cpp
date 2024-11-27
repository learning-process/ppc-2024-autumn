#include "mpi/borisov_s_my_broadcast/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace std::chrono_literals;

namespace borisov_s_my_broadcast {

bool MPITaskMatrixClustering::pre_processing() {
  internal_order_test();
  return true;
}

bool MPITaskMatrixClustering::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.size() < 2 || taskData->inputs_count.size() < 2) {
      return false;
    }

    size_t rows = taskData->inputs_count[0];
    size_t cols = taskData->inputs_count[1];

    if (rows <= 0 || cols <= 0) {
      return false;
    }
  }

  return true;
}

bool MPITaskMatrixClustering::run() {
  internal_order_test();

  size_t rows = 0;
  size_t cols = 0;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
    epsilon_ = *reinterpret_cast<double *>(taskData->inputs[1]);
  }

  my_broadcast(world, rows, 0);
  my_broadcast(world, cols, 0);
  my_broadcast(world, epsilon_, 0);

  matrix_.resize(rows * cols);

  if (world.rank() == 0) {
    auto *data_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
    std::copy(data_ptr, data_ptr + (rows * cols), matrix_.begin());

    for (size_t i = 0; i < rows; i++) {
      double norm = 0.0;
      for (size_t j = 0; j < cols; j++) {
        norm += matrix_[(i * cols) + j] * matrix_[(i * cols) + j];
      }
      norm = std::sqrt(norm);
      for (size_t j = 0; j < cols; j++) {
        matrix_[(i * cols) + j] /= norm;
      }
    }
  }

  my_broadcast(world, matrix_.data(), static_cast<int>(matrix_.size()), 0);

  std::vector<double> global_center(cols, 0.0);

  if (world.rank() == 0) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        global_center[j] += matrix_[(i * cols) + j];
      }
    }
    for (double &val : global_center) {
      val /= static_cast<double>(rows);
    }
  }

  my_broadcast(world, global_center.data(), static_cast<int>(cols), 0);

  size_t rows_per_process = rows / world.size();
  size_t extra_rows = rows % world.size();
  size_t start_row = (world.rank() * rows_per_process) + std::min<size_t>(world.rank(), extra_rows);
  size_t end_row = start_row + rows_per_process + (static_cast<size_t>(world.rank()) < extra_rows ? 1 : 0);

  std::vector<int> local_classification(end_row - start_row, 0);
  for (size_t i = start_row; i < end_row; i++) {
    double distance = 0.0;
    for (size_t j = 0; j < cols; j++) {
      double diff = matrix_[(i * cols) + j] - global_center[j];
      distance += diff * diff;
    }
    distance = std::sqrt(distance);
    local_classification[i - start_row] = (distance < epsilon_) ? 1 : 0;
  }

  int local_size = static_cast<int>(local_classification.size());
  std::vector<int> receive_counts(world.size(), 0);
  std::vector<int> displacements(world.size(), 0);

  boost::mpi::gather(world, local_size, receive_counts, 0);

  if (world.rank() == 0) {
    displacements[0] = 0;
    for (size_t i = 1; i < receive_counts.size(); ++i) {
      displacements[i] = displacements[i - 1] + receive_counts[i - 1];
    }
  }

  std::vector<int> global_classification;
  if (world.rank() == 0) {
    global_classification.resize(rows);
  }

  boost::mpi::gatherv(world, local_classification.data(), local_size, global_classification.data(), receive_counts,
                      displacements, 0);

  if (world.rank() == 0) {
    auto *output_ptr = reinterpret_cast<int *>(taskData->outputs[0]);
    std::copy(global_classification.begin(), global_classification.end(), output_ptr);
  }

  return true;
}

bool MPITaskMatrixClustering::post_processing() {
  internal_order_test();
  return true;
}

}  // namespace borisov_s_my_broadcast
