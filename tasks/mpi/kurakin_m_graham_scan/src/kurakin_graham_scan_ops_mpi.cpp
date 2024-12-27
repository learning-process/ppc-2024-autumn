#include "mpi/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <thread>
#include <vector>

bool kurakin_m_graham_scan_mpi::isLeftAngle(std::vector<double>& p1, std::vector<double>& p2, std::vector<double>& p3) {
  return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) < 0;
}

int kurakin_m_graham_scan_mpi::grahamScan(std::vector<std::vector<double>>& input_) {
  int count_point = input_.size();
  int ind_min_y = std::min_element(input_.begin(), input_.end(),
                                   [&](std::vector<double> a, std::vector<double> b) {
                                     return a[1] < b[1] || (a[1] == b[1] && a[0] > b[0]);
                                   }) -
                  input_.begin();
  std::swap(input_[0], input_[ind_min_y]);
  std::sort(input_.begin() + 1, input_.end(),
            [&](std::vector<double> a, std::vector<double> b) { return isLeftAngle(a, input_[0], b); });

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
  }
  return count_point / size + 1;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  count_point = (int)taskData->inputs_count[0] / 2;
  input_ = std::vector<std::vector<double>>(count_point, std::vector<double>(2, 0));
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (int i = 0; i < count_point * 2; i += 2) {
    input_[i / 2][0] = tmp_ptr[i];
    input_[i / 2][1] = tmp_ptr[i + 1];
  }

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 1 && taskData->inputs_count.size() == 1 && taskData->outputs.size() == 2 &&
         taskData->outputs_count.size() == 2 && taskData->inputs_count[0] % 2 == 0 &&
         taskData->inputs_count[0] / 2 > 2 && taskData->outputs_count[0] == 1 &&
         taskData->inputs_count[0] == taskData->outputs_count[1];
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  count_point = grahamScan(input_);

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = count_point;
  for (int i = 0; i < count_point * 2; i += 2) {
    reinterpret_cast<double*>(taskData->outputs[1])[i] = input_[i / 2][0];
    reinterpret_cast<double*>(taskData->outputs[1])[i + 1] = input_[i / 2][1];
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
    return taskData->inputs.size() == 1 && taskData->inputs_count.size() == 1 && taskData->outputs.size() == 2 &&
           taskData->outputs_count.size() == 2 && taskData->inputs_count[0] % 2 == 0 &&
           taskData->inputs_count[0] / 2 > 2 && taskData->outputs_count[0] == 1 &&
           taskData->inputs_count[0] == taskData->outputs_count[1];
  }

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    count_point = (int)taskData->inputs_count[0] / 2;
    input_ = std::vector<double>((int)taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    for (int i = 0; i < count_point * 2; i += 2) {
      input_[i] = tmp_ptr[i];
      input_[i + 1] = tmp_ptr[i + 1];
    }
  }
  broadcast(world, count_point, 0);

  if (world.rank() == 0) {
    local_count_point = getCountPoint(count_point, world.size(), world.rank());
    local_input_ = std::vector<double>(input_.begin(), input_.begin() + local_count_point * 2);

    int sum_next_count_point = local_count_point;
    for (int i = 1; i < world.size(); i++) {
      int next_count_point = getCountPoint(count_point, world.size(), i);
      if (next_count_point != 0) {
        world.send(i, 0, input_.data() + sum_next_count_point * 2, next_count_point * 2);
      }
      sum_next_count_point += next_count_point;
    }
  } else {
    local_count_point = getCountPoint(count_point, world.size(), world.rank());
    if (local_count_point != 0) {
      local_input_ = std::vector<double>(local_count_point * 2);
      world.recv(0, 0, local_input_.data(), local_count_point * 2);
    }
  }

  if (local_count_point != 0) {
    graham_input_ = std::vector<std::vector<double>>(local_count_point, std::vector<double>(2, 0));
    for (int i = 0; i < local_count_point * 2; i += 2) {
      graham_input_[i / 2][0] = local_input_[i];
      graham_input_[i / 2][1] = local_input_[i + 1];
    }
    local_count_point = grahamScan(graham_input_);
    for (int i = 0; i < local_count_point * 2; i += 2) {
      local_input_[i] = graham_input_[i / 2][0];
      local_input_[i + 1] = graham_input_[i / 2][1];
    }
  }

  if (world.rank() == 0) {
    std::copy(local_input_.begin(), local_input_.begin() + local_count_point * 2, input_.begin());
    int sum_next_count_point = local_count_point;
    for (int i = 1; i < world.size(); i++) {
      int next_count_point;
      world.recv(i, 0, &next_count_point, 1);
      if (next_count_point != 0) {
        world.recv(i, 0, input_.data() + sum_next_count_point * 2, next_count_point * 2);
      }
      sum_next_count_point += next_count_point;
    }
    local_count_point = sum_next_count_point;
  } else {
    world.send(0, 0, &local_count_point, 1);
    if (local_count_point != 0) {
      world.send(0, 0, local_input_.data(), local_count_point * 2);
    }
  }

  if (world.rank() == 0) {
    graham_input_ = std::vector<std::vector<double>>(local_count_point, std::vector<double>(2, 0));
    for (int i = 0; i < local_count_point * 2; i += 2) {
      graham_input_[i / 2][0] = input_[i];
      graham_input_[i / 2][1] = input_[i + 1];
    }
    count_point = grahamScan(graham_input_);
  }

  return true;
}

bool kurakin_m_graham_scan_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = count_point;
    for (int i = 0; i < count_point * 2; i += 2) {
      reinterpret_cast<double*>(taskData->outputs[1])[i] = graham_input_[i / 2][0];
      reinterpret_cast<double*>(taskData->outputs[1])[i + 1] = graham_input_[i / 2][1];
    }
  }

  return true;
}
