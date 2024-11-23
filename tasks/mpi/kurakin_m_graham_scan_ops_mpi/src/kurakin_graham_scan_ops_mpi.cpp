#include "mpi/kurakin_m_graham_scan_ops_mpi/include/kurakin_graham_scan_ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

double kurakin_m_graham_scan_mpi::getRandomDouble(double start = 0.0, double end = 100.0) {
  std::random_device dev;
  std::mt19937 gen(dev());
  double res = (double)(gen() % ((int)(start * 100) - (int)(end * 100)) + (int)(start * 100)) / 100;
  return res;
}

int kurakin_m_graham_scan_mpi::getRandomInt(int start = 0, int end = 100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int res = gen() % (end - start) + start;
  return res;
}

void kurakin_m_graham_scan_mpi::getRandomVectorForGrahamScan(std::vector<double>& res_x, std::vector<double>& res_y,
                                                             int count_point, int size) {
  res_x = std::vector<double>();
  res_y = std::vector<double>();
  for (int i = 0; i < size; i++) {
    int count = getCountPoint(count_point, size, i);
    double start_x = getRandomDouble(-10.0, 10.0);
    double start_y = getRandomDouble(-10.0, 10.0);
    for (int j = 0; j < count; j++) {
      double length = getRandomDouble(0.0, 10.0);
      int angle = getRandomInt(0, 180);
      res_x.push_back(start_x + length * cos(angle));
      res_y.push_back(start_y + length * sin(angle));
    }
  }
}

bool kurakin_m_graham_scan_mpi::isLeftAngle(std::vector<double>& p1, std::vector<double>& p2, std::vector<double>& p3) {
  return ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2])) < 0;
}

int kurakin_m_graham_scan_mpi::grahamScan(std::vector<std::vector<double>>& input_) {
  int count_point = input_.size();
  int ind_min_y = std::min_element(input_.begin(), input_.end(),
                                   [&](std::vector<double> a, std::vector<double> b) {
                                     return a[2] < b[2] || (a[2] == b[2] && a[1] > b[1]);
                                   }) -
                  input_.begin();
  std::swap(input_[0], input_[ind_min_y]);
  for (int i = 1; i < count_point; i++) {
    input_[i][0] =
        (input_[i][1] - input_[0][1]) / sqrt(pow(input_[i][1] - input_[0][1], 2) + pow(input_[i][2] - input_[0][2], 2));
  }
  std::sort(input_.begin() + 1, input_.end(),
            [&](std::vector<double> a, std::vector<double> b) { return a[0] > b[0]; });

  int k = 1;
  for (int i = 2; i < count_point; i++) {
    while (k > 0 && isLeftAngle(input_[k - 1], input_[k], input_[i])) {
      k--;
    }
    std::swap(input_[i], input_[k + 1]);
    k++;
  }
  return k + 1;
}

int kurakin_m_graham_scan_mpi::getCountPoint(int count_point, int size, int rank) {
  if (count_point / size < 3) {
    if (count_point / 3 <= rank) return 0;
    size = count_point / 3;
  }
  if (count_point % size <= rank) {
    return count_point / size;
  } else {
    return count_point / size + 1;
  }
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->inputs.size() == 2 && taskData->inputs_count.size() == 2 && taskData->outputs.size() == 3 &&
         taskData->outputs_count.size() == 3 && taskData->inputs_count[0] > 2 &&
         taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1 &&
         taskData->outputs_count[1] == taskData->outputs_count[2] &&
         taskData->inputs_count[0] == taskData->outputs_count[1];
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  count_point = (int)taskData->inputs_count[0];
  input_ = std::vector<std::vector<double>>(count_point, std::vector<double>(3, 0));
  auto* x_tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* y_tmp_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < count_point; i++) {
    input_[i][1] = x_tmp_ptr[i];
    input_[i][2] = y_tmp_ptr[i];
  }

  int ind_min_y = std::min_element(input_.begin(), input_.end(),
                                   [&](std::vector<double> a, std::vector<double> b) {
                                     return a[2] < b[2] || (a[2] == b[2] && a[1] > b[1]);
                                   }) -
                  input_.begin();
  std::swap(input_[0], input_[ind_min_y]);
  for (int i = 1; i < count_point; i++) {
    input_[i][0] =
        (input_[i][1] - input_[0][1]) / sqrt(pow(input_[i][1] - input_[0][1], 2) + pow(input_[i][2] - input_[0][2], 2));
  }
  std::sort(input_.begin() + 1, input_.end(),
            [&](std::vector<double> a, std::vector<double> b) { return a[0] > b[0]; });

  int k = 1;
  for (int i = 2; i < count_point; i++) {
    while (k > 0 && isLeftAngle(input_[k - 1], input_[k], input_[i])) {
      k--;
    }
    std::swap(input_[i], input_[k + 1]);
    k++;
  }
  count_point = k + 1;

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = count_point;
  for (int i = 0; i < count_point; i++) {
    reinterpret_cast<double*>(taskData->outputs[1])[i] = input_[i][1];
    reinterpret_cast<double*>(taskData->outputs[2])[i] = input_[i][2];
  }

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->inputs.size() == 2 && taskData->inputs_count.size() == 2 && taskData->outputs.size() == 3 &&
           taskData->outputs_count.size() == 3 && taskData->inputs_count[0] > 2 &&
           taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1 &&
           taskData->outputs_count[1] == taskData->outputs_count[2] &&
           taskData->inputs_count[0] == taskData->outputs_count[1];
  }

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    count_point = (int)taskData->inputs_count[0];
    input_x_ = std::vector<double>(count_point);
    input_y_ = std::vector<double>(count_point);
    auto* x_tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* y_tmp_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
    for (int i = 0; i < count_point; i++) {
      input_x_[i] = x_tmp_ptr[i];
      input_y_[i] = y_tmp_ptr[i];
    }
  }
  broadcast(world, count_point, 0);

  if (world.rank() == 0) {
    local_count_point = getCountPoint(count_point, world.size(), world.rank());
    local_input_x_ = std::vector<double>(input_x_.begin(), input_x_.begin() + local_count_point);
    local_input_y_ = std::vector<double>(input_y_.begin(), input_y_.begin() + local_count_point);
    
    int sum_next_count_point = local_count_point;
    for (int i = 1; i < world.size(); i++) {
      int next_count_point = getCountPoint(count_point, world.size(), i);
      if (next_count_point != 0) {
        world.send(i, 0, input_x_.data() + sum_next_count_point, next_count_point);
        world.send(i, 0, input_y_.data() + sum_next_count_point, next_count_point);
      }
      sum_next_count_point += next_count_point;
    }
  } else {
    local_count_point = getCountPoint(count_point, world.size(), world.rank());
    if (local_count_point != 0) {
      local_input_x_ = std::vector<double>(local_count_point);
      local_input_y_ = std::vector<double>(local_count_point);
      world.recv(0, 0, local_input_x_.data(), local_count_point);
      world.recv(0, 0, local_input_y_.data(), local_count_point);
    }
  }

  if (local_count_point != 0) {
    local_input_ = std::vector<std::vector<double>>(local_count_point, std::vector<double>(3, 0));
    for (int i = 0; i < local_count_point; i++) {
      local_input_[i][1] = local_input_x_[i];
      local_input_[i][2] = local_input_y_[i];
    }
    local_count_point = grahamScan(local_input_);
    for (int i = 0; i < local_count_point; i++) {
      local_input_x_[i] = local_input_[i][1];
      local_input_y_[i] = local_input_[i][2];
    }
  }

  if (world.rank() == 0) {
    std::copy(local_input_x_.begin(), local_input_x_.begin() + local_count_point, input_x_.begin());
    std::copy(local_input_y_.begin(), local_input_y_.begin() + local_count_point, input_y_.begin());
    int sum_next_count_point = local_count_point;
    for (int i = 1; i < world.size(); i++) {
      int next_count_point;
      world.recv(i, 0, &next_count_point, 1);
      if (next_count_point != 0) {
        world.recv(i, 0, input_x_.data() + sum_next_count_point, next_count_point);
        world.recv(i, 0, input_y_.data() + sum_next_count_point, next_count_point);
      }
      sum_next_count_point += next_count_point;
    }
    local_count_point = sum_next_count_point;
  } else {
    world.send(0, 0, &local_count_point, 1);
    if (local_count_point != 0) {
      world.send(0, 0, local_input_x_.data(), local_count_point);
      world.send(0, 0, local_input_y_.data(), local_count_point);
    }
  }

  if (world.rank() == 0) {
    local_input_ = std::vector<std::vector<double>>(local_count_point, std::vector<double>(3, 0));
    for (int i = 0; i < local_count_point; i++) {
      local_input_[i][1] = input_x_[i];
      local_input_[i][2] = input_y_[i];
    }
    count_res_point = grahamScan(local_input_);
  }

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = count_res_point;
    for (int i = 0; i < count_res_point; i++) {
      reinterpret_cast<double*>(taskData->outputs[1])[i] = local_input_[i][1];
      reinterpret_cast<double*>(taskData->outputs[2])[i] = local_input_[i][2];
    }
  }

  return true;
}
