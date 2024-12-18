#include "seq/beskhmelnova_k_jarvis_march/include/jarvis_march.hpp"

#include <random>

using namespace std::chrono_literals;

template <typename DataType>
DataType beskhmelnova_k_jarvis_march_seq::crossProduct(const std::vector<DataType>& p1, const std::vector<DataType>& p2,
                                                       const std::vector<DataType>& p3) {
  return ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2]));
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_seq::isLeftAngle(const std::vector<DataType>& p1, const std::vector<DataType>& p2,
                                                  const std::vector<DataType>& p3) {
  return cross_product(p1, p2, p3) > 0;
}

template <typename DataType>
void beskhmelnova_k_jarvis_march_seq::jarvisMarch(int& num_points, std::vector<std::vector<DataType>>& input,
                                                  std::vector<DataType>& res_x, std::vector<DataType>& res_y) {
  int leftmost_index = 0;
  for (int i = 1; i < num_points; i++)
    if (input[i][1] < input[leftmost_index][1] ||
        (input[i][1] == input[leftmost_index][1] && input[i][2] < input[leftmost_index][2]))
      leftmost_index = i;
  std::swap(input[0], input[leftmost_index]);
  std::vector<std::vector<DataType>> hull;
  int current = 0;
  do {
    hull.push_back(input[current]);
    int next = (current + 1) % num_points;
    for (int i = 0; i < num_points; ++i) {
      auto cross_product =
          beskhmelnova_k_jarvis_march_seq::crossProduct<DataType>(input[current], input[next], input[i]);
      if (cross_product < 0 || (cross_product == 0 &&
                                std::hypot(input[i][1] - input[current][1], input[i][2] - input[current][2]) >
                                    std::hypot(input[next][1] - input[current][1], input[next][2] - input[current][2])))
        next = i;
    }
    current = next;
  } while (current != 0);
  num_points = static_cast<int>(hull.size());
  res_x.resize(num_points);
  res_y.resize(num_points);
  for (int i = 0; i < num_points; ++i) {
    res_x[i] = hull[i][1];
    res_y[i] = hull[i][2];
  }
  input = hull;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_seq::TestTaskSequential<DataType>::pre_processing() {
  internal_order_test();
  num_points = (int)taskData->inputs_count[0];
  input = std::vector<std::vector<DataType>>(num_points, std::vector<DataType>(3, 0));
  auto* ptr_x = reinterpret_cast<DataType*>(taskData->inputs[0]);
  auto* ptr_y = reinterpret_cast<DataType*>(taskData->inputs[1]);
  for (int i = 0; i < num_points; i++) {
    input[i][1] = ptr_x[i];
    input[i][2] = ptr_y[i];
  }
  return true;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_seq::TestTaskSequential<DataType>::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 3 && taskData->outputs_count.size() == 3;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_seq::TestTaskSequential<DataType>::run() {
  internal_order_test();
  jarvisMarch(num_points, input, res_x, res_y);
  return true;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_seq::TestTaskSequential<DataType>::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = num_points;
  for (int i = 0; i < num_points; i++) {
    reinterpret_cast<DataType*>(taskData->outputs[1])[i] = input[i][1];
    reinterpret_cast<DataType*>(taskData->outputs[2])[i] = input[i][2];
  }
  return true;
}

template class beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double>;