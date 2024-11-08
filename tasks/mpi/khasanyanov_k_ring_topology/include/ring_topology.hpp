#ifndef _Ring_TOPOLOGY_HPP_
#define _Ring_TOPOLOGY_HPP_

#include <concepts>
#include <cstddef>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace khasanyanov_k_ring_topology_mpi {

template <typename T>
concept Copyable = requires(const T& a) {
  T{a};
};

template <typename T>
concept Unsigned = (bool)std::is_unsigned<T>();

template <Copyable DataType, Unsigned SizeType = std::uint32_t>
class RingTopology : ppc::core::Task {
  static_assert(sizeof(SizeType) <= sizeof(std::uint32_t),
                "Size of 'SizeType' greater than std::uint32_t, possible loss of data");

 private:
  boost::mpi::communicator world;
  std::vector<DataType> start_data_, end_data_;
  std::shared_ptr<std::vector<int>> order_;

  enum Tags { Default, Size, Data };

 public:
  explicit RingTopology(std::shared_ptr<ppc::core::TaskData> taskData_, std::shared_ptr<std::vector<int>>& order)
      : Task(std::move(taskData_)), order_(order) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  [[nodiscard]] static std::vector<int> true_order(int);
};

template <Copyable DataType, Unsigned SizeType>
bool RingTopology<DataType, SizeType>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return !taskData->inputs.empty() && !taskData->inputs_count.empty() && taskData->inputs_count[0] > 0;
  }
  return true;
}

template <Copyable DataType, Unsigned SizeType>
bool RingTopology<DataType, SizeType>::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp_data = reinterpret_cast<DataType*>(taskData->inputs[0]);
    auto tmp_size = static_cast<SizeType>(taskData->inputs_count[0]);
    start_data_.assign(tmp_data, tmp_data + tmp_size);
    end_data_.reserve(tmp_size);
  }
  return true;
}

template <Copyable DataType, Unsigned SizeType>
bool RingTopology<DataType, SizeType>::run() {
  internal_order_test();
  auto rank = world.rank();
  int next = (rank == world.size() - 1) ? 0 : rank + 1;
  int prev = (rank == world.size() - 1) ? world.size() - 1 : rank - 1;
  size_t size;
  if (rank == 0) {
    size = start_data_.size();
    world.send(next, Size, size);
    world.send(next, Data, start_data_.data(), size);

    world.recv(prev, Data, end_data_.data(), size);
  } else {
    world.recv(prev, Size, size);
    world.recv(prev, Data, start_data_.data(), size);

    order_->push_back(rank);

    if (rank != world.size() - 1) {
      world.send(next, Size, size);
    }
    world.send(next, Data, start_data_.data(), size);
  }
  return true;
}

template <Copyable DataType, Unsigned SizeType>
bool RingTopology<DataType, SizeType>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(end_data_.data()));
    taskData->outputs_count.emplace_back(end_data_.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(order_->data()));
    taskData->outputs_count.emplace_back(order_->size());
  }
  return true;
}

template <Copyable DataType, Unsigned SizeType>
std::vector<int> RingTopology<DataType, SizeType>::true_order(int num_processes) {
  std::vector<int> true_order(num_processes);
  for (int i = 0; i < num_processes - 1; ++i) {
    true_order[i] = i + 1;
  }
  true_order[num_processes] = 0;
  return true_order;
}

}  // namespace khasanyanov_k_ring_topology_mpi

#endif