// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <cmath>
#include <stack>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace matyunina_a_batcher_qsort_mpi {

template <typename T>
void quickSort(std::vector<T>& data);

template <typename T>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> data_;
};

template <typename T>
class TestTaskParallel : public ppc::core::Task {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> global_;
  std::vector<T> local_;
  boost::mpi::communicator world;
};

}  // namespace matyunina_a_batcher_qsort_mpi

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskSequential<T>::pre_processing() {
  internal_order_test();

  auto* input = reinterpret_cast<T*>(taskData->inputs[0]);
  data_.assign(input, input + taskData->inputs_count[0]);

  return true;
}

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskSequential<T>::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == taskData->outputs_count[0]);
}

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskSequential<T>::run() {
  internal_order_test();

  quickSort(data_);

  return true;
}

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskSequential<T>::post_processing() {
  internal_order_test();

  std::copy(data_.begin(), data_.end(), reinterpret_cast<T*>(taskData->outputs[0]));

  return true;
}

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskParallel<T>::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* input = reinterpret_cast<T*>(taskData->inputs[0]);
    global_.assign(input, input + taskData->inputs_count[0]);
  }

  return true;
}

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskParallel<T>::validation() {
  internal_order_test();
  return (world.rank() != 0) || (taskData->inputs_count[0] == taskData->outputs_count[0]);
}

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskParallel<T>::run() {
  internal_order_test();

  int32_t data_size;
  if (world.rank() == 0) data_size = global_.size();
  broadcast(world, data_size, 0);

  if (data_size == 0) {
    return true;
  }

  uint32_t delta = data_size / world.size();
  uint32_t remainder = data_size % world.size();
  uint32_t local_size = delta + (((uint32_t)world.rank() < remainder) ? 1 : 0);

  std::vector<boost::mpi::request> reqs;
  if (world.rank() == 0) {
    local_.assign(global_.begin(), global_.begin() + local_size);
    uint32_t offset = local_size;
    uint32_t send_size;
    for (int32_t proc = 1; proc < world.size(); proc++) {
      send_size = delta + (((uint32_t)proc < remainder) ? 1 : 0);
      reqs.emplace_back(world.isend(proc, proc, global_.data() + offset, send_size));
      offset += send_size;
    }
  } else {
    local_.resize(local_size);
    world.irecv(0, world.rank(), local_.data(), local_size).wait();
  }

  quickSort(local_);

  if (world.rank() == 0) {
    for (auto& req : reqs) {
      req.wait();
    }
  }

  auto merge = [](std::vector<T>& first, std::vector<T>& second) {
    std::vector<T> temp(first.size() + second.size());
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;

    while (i < first.size() && j < second.size()) {
      temp[k++] = (first[i] <= second[j]) ? first[i++] : second[j++];
    }
    while (i < first.size()) temp[k++] = first[i++];
    while (j < second.size()) temp[k++] = second[j++];

    for (size_t idx = 0; idx < first.size(); idx++) {
      first[idx] = temp[idx];
    }
    for (size_t idx = 0; idx < second.size(); idx++) {
      second[idx] = temp[first.size() + idx];
    }
  };

  int32_t offset = -1;
  uint32_t proc_size;
  std::vector<T> temp;
  int32_t merge_count = std::pow(std::ceil(log2(data_size)), 2);
  for (int32_t i = 0; i <= merge_count; i++) {
    if (world.rank() % 2 == i % 2) {
      if (world.rank() - offset >= world.size() || world.rank() - offset < 0) {
        continue;
      }
      proc_size = delta + (world.rank() - offset < (int32_t)remainder ? 1 : 0);
      temp.resize(proc_size);
      world.irecv(world.rank() - offset, 0, temp.data(), proc_size).wait();
      merge(local_, temp);
      world.isend(world.rank() - offset, 1, temp.data(), temp.size()).wait();
    } else {
      if (world.rank() + offset >= world.size() || world.rank() + offset < 0) {
        continue;
      }
      auto req = world.isend(world.rank() + offset, 0, local_.data(), local_.size());
      world.irecv(world.rank() + offset, 1, local_.data(), local_.size()).wait();
      req.wait();
    }
  }

  if (world.rank() == 0) {
    reqs.clear();
    std::copy(local_.begin(), local_.end(), global_.begin());

    uint32_t global_offset = local_.size();
    uint32_t recv_size;
    for (int32_t proc = 1; proc < world.size(); proc++) {
      recv_size = delta + (((uint32_t)proc < remainder) ? 1 : 0);
      reqs.emplace_back(world.irecv(proc, proc, global_.data() + global_offset, recv_size));
      global_offset += recv_size;
    }

    for (auto& req : reqs) {
      req.wait();
    }
  } else {
    world.isend(0, world.rank(), local_.data(), local_.size()).wait();
  }

  return true;
}

template <typename T>
bool matyunina_a_batcher_qsort_mpi::TestTaskParallel<T>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(global_.begin(), global_.end(), reinterpret_cast<T*>(taskData->outputs[0]));
  }

  return true;
}

template <typename T>
void matyunina_a_batcher_qsort_mpi::quickSort(std::vector<T>& data) {
  if (data.empty()) return;

  std::stack<std::pair<uint32_t, uint32_t>> stack;
  stack.emplace(0, data.size() - 1);

  uint32_t low;
  uint32_t high;
  T pivot;
  uint32_t i;
  std::pair<uint32_t, uint32_t> range;
  while (!stack.empty()) {
    range = stack.top();
    stack.pop();

    low = range.first;
    high = range.second;
    if (low < high) {
      pivot = data[high];
      i = low;

      for (uint32_t j = low; j < high; j++) {
        if (data[j] <= pivot) {
          if (i != j) {
            std::swap(data[i], data[j]);
          }
          i++;
        }
      }
      std::swap(data[i], data[high]);

      if (i > 0 && low < i - 1) {
        stack.emplace(low, i - 1);
      }
      if (i + 1 < high) {
        stack.emplace(i + 1, high);
      }
    }
  }
}
