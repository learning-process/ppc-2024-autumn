#include "seq/borisov_s_broadcast/include/ops_seq.hpp"

#include <random>
#include <thread>

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
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
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
  auto* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (size_t i = 0; i < distance_matrix_.size(); i++) {
    output_ptr[i] = distance_matrix_[i];
  }
  return true;
}

}  // namespace borisov_s_broadcast