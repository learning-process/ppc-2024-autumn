#include "mpi/beskhmelnova_k_jarvis_march/include/jarvis_march.hpp"

#include <random>

using namespace std::chrono_literals;

int beskhmelnova_k_jarvis_march_mpi::localNumPoints(int num_points, int world_size, int rank) {
  if (num_points / world_size < 3) {
    if (num_points / 3 <= rank) return 0;
    world_size = num_points / 3;
  }
  if (num_points % world_size <= rank) return num_points / world_size;
  return num_points / world_size + 1;
}

template <typename DataType>
DataType beskhmelnova_k_jarvis_march_mpi::crossProduct(const std::vector<DataType>& p1, const std::vector<DataType>& p2,
                                                       const std::vector<DataType>& p3) {
  return ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2]));
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_mpi::isLeftAngle(const std::vector<DataType>& p1, const std::vector<DataType>& p2,
                                                  const std::vector<DataType>& p3) {
  return cross_product(p1, p2, p3) > 0;
}

template <typename DataType>
void beskhmelnova_k_jarvis_march_mpi::jarvisMarch(int& num_points, std::vector<std::vector<DataType>>& input,
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
          beskhmelnova_k_jarvis_march_mpi::crossProduct<DataType>(input[current], input[next], input[i]);
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
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<DataType>::pre_processing() {
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
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<DataType>::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 3 && taskData->outputs_count.size() == 3;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<DataType>::run() {
  internal_order_test();
  jarvisMarch(num_points, input, res_x, res_y);
  return true;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<DataType>::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = num_points;
  for (int i = 0; i < num_points; i++) {
    reinterpret_cast<DataType*>(taskData->outputs[1])[i] = input[i][1];
    reinterpret_cast<DataType*>(taskData->outputs[2])[i] = input[i][2];
  }
  return true;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<DataType>::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    num_points = (int)taskData->inputs_count[0];
    input_x = std::vector<DataType>(num_points);
    input_y = std::vector<DataType>(num_points);
    auto* ptr_x = reinterpret_cast<DataType*>(taskData->inputs[0]);
    auto* ptr_y = reinterpret_cast<DataType*>(taskData->inputs[1]);
    for (int i = 0; i < num_points; i++) {
      input_x[i] = ptr_x[i];
      input_y[i] = ptr_y[i];
    }
  }
  return true;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<DataType>::validation() {
  internal_order_test();
  if (world.rank() == 0) return taskData->inputs_count[0] >= 3 && taskData->outputs_count.size() == 3;
  return true;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<DataType>::run() {
  internal_order_test();
  broadcast(world, num_points, 0);
  if (world.rank() == 0) {
    local_num_points = localNumPoints(num_points, world.size(), world.rank());
    local_input_x = std::vector<DataType>(input_x.begin(), input_x.begin() + local_num_points);
    local_input_y = std::vector<DataType>(input_y.begin(), input_y.begin() + local_num_points);
    int sum_next_num_points = local_num_points;
    for (int i = 1; i < world.size(); i++) {
      int next_num_points = localNumPoints(num_points, world.size(), i);
      if (next_num_points != 0) {
        world.send(i, 0, input_x.data() + sum_next_num_points, next_num_points);
        world.send(i, 0, input_y.data() + sum_next_num_points, next_num_points);
      }
      sum_next_num_points += next_num_points;
    }
  } else {
    local_num_points = localNumPoints(num_points, world.size(), world.rank());
    if (local_num_points != 0) {
      local_input_x = std::vector<DataType>(local_num_points);
      local_input_y = std::vector<DataType>(local_num_points);
      world.recv(0, 0, local_input_x.data(), local_num_points);
      world.recv(0, 0, local_input_y.data(), local_num_points);
    }
  }

  if (local_num_points != 0) {
    local_input = std::vector<std::vector<DataType>>(local_num_points, std::vector<DataType>(3, 0));
    for (int i = 0; i < local_num_points; i++) {
      local_input[i][1] = local_input_x[i];
      local_input[i][2] = local_input_y[i];
    }
    std::vector<DataType> local_res_x;
    std::vector<DataType> local_res_y;
    jarvisMarch(local_num_points, local_input, local_res_x, local_res_y);
    local_input_x = local_res_x;
    local_input_y = local_res_y;
  }
  if (world.rank() == 0) {
    input_x.assign(local_input_x.begin(), local_input_x.end());
    input_y.assign(local_input_y.begin(), local_input_y.end());
    int sum_next_num_points = local_num_points;
    for (int i = 1; i < world.size(); i++) {
      int next_num_points;
      world.recv(i, 0, &next_num_points, 1);
      if (next_num_points != 0) {
        world.recv(i, 0, input_x.data() + sum_next_num_points, next_num_points);
        world.recv(i, 0, input_y.data() + sum_next_num_points, next_num_points);
      }
      sum_next_num_points += next_num_points;
    }
    local_num_points = sum_next_num_points;
  } else {
    world.send(0, 0, &local_num_points, 1);
    if (local_num_points != 0) {
      world.send(0, 0, local_input_x.data(), local_num_points);
      world.send(0, 0, local_input_y.data(), local_num_points);
    }
  }
  if (world.rank() == 0) {
    local_input = std::vector<std::vector<DataType>>(local_num_points, std::vector<DataType>(3, 0));
    for (int i = 0; i < local_num_points; i++) {
      local_input[i][1] = input_x[i];
      local_input[i][2] = input_y[i];
    }
    std::vector<DataType> final_res_x;
    std::vector<DataType> final_res_y;
    jarvisMarch(local_num_points, local_input, final_res_x, final_res_y);
    res_x = final_res_x;
    res_y = final_res_y;
  }
  return true;
}

template <typename DataType>
bool beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<DataType>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = static_cast<int>(res_x.size());
    for (size_t i = 0; i < res_x.size(); i++) {
      reinterpret_cast<DataType*>(taskData->outputs[1])[i] = res_x[i];
      reinterpret_cast<DataType*>(taskData->outputs[2])[i] = res_y[i];
    }
  }
  return true;
}

template class beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<double>;
template class beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double>;

template class beskhmelnova_k_jarvis_march_mpi::TestMPITaskSequential<int>;
template class beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<int>;
