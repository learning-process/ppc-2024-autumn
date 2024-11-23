#include "mpi/borisov_s_broadcast/include/ops_mpi.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace std::chrono_literals;

namespace borisov_s_broadcast {

std::vector<double> getRandomPoints(int count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(0.0, 100.0);

  std::vector<double> points(count * 2);
  for (int i = 0; i < count * 2; i++) {
    points[i] = dist(gen);
  }
  return points;
}

bool DistanceMatrixTaskSequential::pre_processing() {
  internal_order_test();

  size_t points_count = taskData->inputs_count[0];
  points_.resize(points_count * 2);
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  for (size_t i = 0; i < points_count * 2; i++) {
    points_[i] = tmp_ptr[i];
  }

  distance_matrix_.resize(points_count * points_count, 0.0);
  return true;
}

bool DistanceMatrixTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs[0] != nullptr && taskData->outputs[0] != nullptr;
}

bool DistanceMatrixTaskSequential::run() {
  internal_order_test();
  size_t points_count = taskData->inputs_count[0];

  for (size_t i = 0; i < points_count; i++) {
    for (size_t j = i + 1; j < points_count; j++) {
      double dx = points_[2 * i] - points_[2 * j];
      double dy = points_[(2 * i) + 1] - points_[(2 * j) + 1];
      double dist = std::sqrt((dx * dx) + (dy * dy));
      distance_matrix_[(i * points_count) + j] = dist;
      distance_matrix_[(j * points_count) + i] = dist;
    }
  }
  return true;
}

bool DistanceMatrixTaskSequential::post_processing() {
  internal_order_test();
  auto *output_ptr = reinterpret_cast<double *>(taskData->outputs[0]);
  for (size_t i = 0; i < distance_matrix_.size(); i++) {
    output_ptr[i] = distance_matrix_[i];
  }
  return true;
}

bool DistanceMatrixTaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool DistanceMatrixTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs[0] != nullptr && taskData->outputs[0] != nullptr;
  }
  return true;
}

bool DistanceMatrixTaskParallel::run() {
  internal_order_test();

  int points_count = 0;
  if (world.rank() == 0) {
    points_count = static_cast<int>(taskData->inputs_count[0]);
  }

  boost::mpi::broadcast(world, points_count, 0);

  size_t points_size = static_cast<size_t>(points_count) * 2;

  if (world.rank() == 0) {
    points_.resize(points_size);
    auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
    for (size_t i = 0; i < points_size; i++) {
      points_[i] = tmp_ptr[i];
    }
  } else {
    points_.resize(points_size);
  }

  if (points_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("points_size is too large");
  }

  int broadcast_size = static_cast<int>(points_size);

  boost::mpi::broadcast(world, points_.data(), broadcast_size, 0);

  size_t base_size = points_count / world.size();
  size_t remainder = points_count % world.size();

  size_t local_start;
  size_t local_end;
  if (static_cast<size_t>(world.rank()) < remainder) {
    local_start = world.rank() * (base_size + 1);
    local_end = local_start + base_size + 1;
  } else {
    local_start = remainder * (base_size + 1) + (world.rank() - remainder) * base_size;
    local_end = local_start + base_size;
  }

  size_t local_count = local_end - local_start;

  std::vector<double> local_distances(points_count * local_count);
  for (size_t i = local_start; i < local_end; i++) {
    if (points_count < 0) {
      throw std::invalid_argument("points_count must be non-negative");
    }
    for (size_t j = 0; j < static_cast<size_t>(points_count); j++) {
      double dx = points_[2 * i] - points_[2 * j];
      double dy = points_[(2 * i) + 1] - points_[(2 * j) + 1];
      local_distances[((i - local_start) * points_count) + j] = std::sqrt((dx * dx) + (dy * dy));
    }
  }

  size_t local_size = local_distances.size();
  std::vector<size_t> recv_counts;
  if (world.rank() == 0) {
    recv_counts.resize(world.size());
  }
  boost::mpi::gather(world, local_size, recv_counts, 0);

  if (local_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("local_size is too large");
  }

  int local_size_int = static_cast<int>(local_size);
  std::vector<int> recv_counts_int;
  if (world.rank() == 0) {
    recv_counts_int.resize(world.size());
    for (size_t i = 0; i < recv_counts.size(); i++) {
      recv_counts_int[i] = static_cast<int>(recv_counts[i]);
    }
  }

  std::vector<int> displs_int;
  std::vector<double> all_distances;
  if (world.rank() == 0) {
    displs_int.resize(world.size());
    displs_int[0] = 0;
    for (int i = 1; i < world.size(); i++) {
      displs_int[i] = displs_int[i - 1] + recv_counts_int[i - 1];
    }

    int total_size = displs_int[world.size() - 1] + recv_counts_int[world.size() - 1];
    all_distances.resize(static_cast<size_t>(total_size));
  }

  boost::mpi::gatherv(world, local_distances.data(), local_size_int, all_distances.data(), recv_counts_int, displs_int,
                      0);

  if (world.rank() == 0) {
    distance_matrix_.resize(static_cast<size_t>(points_count) * points_count);
    size_t offset = 0;
    for (int i = 0; i < world.size(); i++) {
      size_t start;
      size_t count;
      if (static_cast<size_t>(i) < remainder) {
        start = static_cast<size_t>(i) * (base_size + 1);
        count = base_size + 1;
      } else {
        start = remainder * (base_size + 1) + (static_cast<size_t>(i) - remainder) * base_size;
        count = base_size;
      }

      for (size_t j = 0; j < count; j++) {
        for (size_t k = 0; k < static_cast<size_t>(points_count); k++) {
          distance_matrix_[((start + j) * points_count) + k] = all_distances[offset + (j * points_count) + k];
        }
      }
      offset += static_cast<size_t>(recv_counts_int[i]);
    }
  }

  return true;
}

bool DistanceMatrixTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto *output_ptr = reinterpret_cast<double *>(taskData->outputs[0]);
    for (size_t i = 0; i < distance_matrix_.size(); i++) {
      output_ptr[i] = distance_matrix_[i];
    }
  }
  return true;
}

}  // namespace borisov_s_broadcast
