#ifndef _Ring_TOPOLOGY_HPP_
#define _Ring_TOPOLOGY_HPP_

#include <concepts>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace khasanyanov_k_ring_topology_mpi {

// not included 'right' border with integers, not included 'left' border always
template <typename T = int>
std::vector<T> generate_random_vector(size_t size = 100, const T& left = T{-1000}, const T& right = T{1000}) {
  std::vector<T> res(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  double frac = (gen() % 100) / 100.0;
  for (size_t i = 0; i < size; i++) {
    res[i] = left + frac + static_cast<T>(gen() % static_cast<int>(right - left));
  }
  return res;
}

template <typename T>
concept Copyable = requires(const T& a) {
  T{a};
};

template <typename T>
concept Unsigned = std::is_unsigned<T>();

template <Copyable DataType, Unsigned SizeType = std::uint32_t>
class RingTopology : ppc::core::Task {
  static_assert(sizeof(SizeType) > sizeof(std::uint32_t),
                "Size of 'SizeType' greater than std::uint32_t, possible loss of data");

 private:
  boost::mpi::communicator world;
  std::vector<DataType> start_data_, end_data_;
  std::shared_ptr<std::vector<int>> order_;

  enum Tags { Default, Size, Data };

 public:
  RingTopology(std::shared_ptr<ppc::core::TaskData> taskData_, std::shared_ptr<std::vector<int>>& order)
      : Task(std::move(taskData_)), order_(order) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  [[nodiscard]] static std::vector<int> true_order(size_t);
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
    auto* tmp_size = reinterpret_cast<SizeType*>(taskData->inputs_count[0]);
    start_data_.assign(tmp_data, tmp_data + *tmp_size);
  }
  return true;
}

template <Copyable DataType, Unsigned SizeType>
bool RingTopology<DataType, SizeType>::run() {
  internal_order_test();
  auto rank = world.rank();
  size_t size;
  if (rank == 0) {
    size = start_data_.size();
    world.send(rank + 1, Size, size);
    world.send(rank + 1, Data, start_data_.data(), size);

    world.recv(world.size() - 1, Data, end_data_.data(), size);
  } else {
    world.recv(rank - 1, Size, size);
    world.recv(rank - 1, Data, start_data_.data(), size);

    order_->push_back(rank);

    if (rank != world.size() - 1) {
      world.send(rank + 1, Size, size);
      world.send(rank + 1, Data, start_data_.data(), size);
    } else {
      world.send(0, Data, start_data_.data(), size);
    }
  }
}

template <Copyable DataType, Unsigned SizeType>
bool RingTopology<DataType, SizeType>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(end_data_.data()));
    taskData->outputs_count.emplace_back(end_data_.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(order_->data()));
  }
  return true;
}

template <Copyable DataType, Unsigned SizeType>
std::vector<int> RingTopology<DataType, SizeType>::true_order(size_t num_processes) {
  std::vector<int> true_order(num_processes);
  for (size_t i = 0; i < num_processes - 1;) {
    true_order[i] = ++i;
  }
  true_order[num_processes] = 0;
  return true_order;
}

}  // namespace khasanyanov_k_ring_topology_mpi

#endif