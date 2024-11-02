// Copyright 2023 Nesterov Alexander
#include "mpi/koshkin_m_scalar_product_of_vectors/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
static int offset = 0;

bool koshkin_m_scalar_product_of_vectors::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<std::vector<int>>(taskData->inputs.size());
  for (size_t i = 0; i < input_.size(); i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    input_[i] = std::vector<int>(taskData->inputs_count[i]);
    for (size_t j = 0; j < taskData->inputs_count[i]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res = 0;
  return true;
}

bool koshkin_m_scalar_product_of_vectors::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs.size() == 2 && taskData->inputs.size() == taskData->inputs_count.size() &&
      taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs.size() == 1 &&
      taskData->outputs.size() == taskData->outputs_count.size() && taskData->outputs_count[0] == 1) {
    return true;
  }
  return false;
}

bool koshkin_m_scalar_product_of_vectors::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_[0].size(); i++) {
    res += input_[0][i] * input_[1][i];
  }
  return true;
}

bool koshkin_m_scalar_product_of_vectors::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool koshkin_m_scalar_product_of_vectors::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  size_t total_el = 0;
  size_t base_el = 0;
  size_t extra_el = 0;
  if (world.rank() == 0) {
    total_el = taskData->inputs_count[0];
    count_processes_ = world.size();
    base_el = total_el / count_processes_;
    extra_el = total_el % count_processes_;
  }
  boost::mpi::broadcast(world, count_processes_, 0);
  counts_.resize(count_processes_);

  if (world.rank() == 0) {
    counts_.assign(count_processes_, base_el);
    std::fill(counts_.begin(), counts_.begin() + extra_el, base_el + 1);
  }
  boost::mpi::broadcast(world, counts_.data(), count_processes_, 0);

  if (world.rank() == 0) {
    input_ = std::vector<std::vector<int>>(taskData->inputs.size());
    for (size_t i = 0; i < input_.size(); i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i] = std::vector<int>(taskData->inputs_count[i]);
      for (size_t j = 0; j < taskData->inputs_count[i]; j++) {
        input_[i][j] = tmp_ptr[j];
      }
    }
  }
  res = 0;
  return true;
}

bool koshkin_m_scalar_product_of_vectors::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs.size() == 2 && taskData->inputs.size() == taskData->inputs_count.size() &&
            taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs.size() == 1 &&
            taskData->outputs.size() == taskData->outputs_count.size() && taskData->outputs_count[0] == 1);
  }
  return true;
}

bool koshkin_m_scalar_product_of_vectors::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.rank() == 0) {
    size_t offset_extra = counts_[0];
    for (unsigned int proces = 1; proces < count_processes_; proces++) {
      size_t cur_accout = counts_[proces];
      world.send(proces, 0, input_[0].data() + offset_extra, cur_accout);
      world.send(proces, 1, input_[1].data() + offset_extra, cur_accout);
      offset_extra += cur_accout;
    }
  }

  local_input1_ = std::vector<int>(counts_[world.rank()]);
  local_input2_ = std::vector<int>(counts_[world.rank()]);

  if (world.rank() > 0) {
    world.recv(0, 0, local_input1_.data(), counts_[world.rank()]);
    world.recv(0, 1, local_input2_.data(), counts_[world.rank()]);
  } else {
    local_input1_ = std::vector<int>(input_[0].begin(), input_[0].begin() + counts_[0]);
    local_input2_ = std::vector<int>(input_[1].begin(), input_[1].begin() + counts_[0]);
  }

  int local_res = 0;

  for (size_t i = 0; i < local_input1_.size(); i++) {
    local_res += local_input1_[i] * local_input2_[i];
  }
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  return true;
}

bool koshkin_m_scalar_product_of_vectors::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

int koshkin_m_scalar_product_of_vectors::calculateDotProduct(const std::vector<int>& vec_1,
                                                             const std::vector<int>& vec_2) {
  long result = 0;
  for (size_t i = 0; i < vec_1.size(); i++) result += vec_1[i] * vec_2[i];
  return result;
}
