#ifndef _RING_TOPOLOGY_HPP_
#define _RING_TOPOLOGY_HPP_

#include <algorithm>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <concepts>
#include <memory>
#include <numeric>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace khasanyanov_k_ring_topology_mpi {

template <std::copyable DataType, std::unsigned_integral SizeType = std::uint32_t>
class RingTopology : public ppc::core::Task {
  static_assert(sizeof(SizeType) <= sizeof(std::uint32_t),
                "Size of 'SizeType' greater than std::uint32_t, possible loss of data");

 private:
  struct Data {
    std::vector<DataType> input_;
    std::vector<int> order_;
    template <class Archive>
    void serialize(Archive& ar, unsigned int version) {
      ar & input_;
      ar & order_;
    }
  } data_;

  boost::mpi::communicator world;
  enum Tags { Default, Data };

 public:
  explicit RingTopology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  [[nodiscard]] static std::vector<int> true_order(int);
};

template <std::copyable DataType, std::unsigned_integral SizeType>
bool RingTopology<DataType, SizeType>::validation() {
  internal_order_test();

  return world.rank() != 0 || (taskData->inputs.size() == 1 && !taskData->inputs_count.empty() &&
                               taskData->inputs_count[0] > 0 && taskData->outputs.size() == 2);
}

template <std::copyable DataType, std::unsigned_integral SizeType>
bool RingTopology<DataType, SizeType>::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp_data = reinterpret_cast<DataType*>(taskData->inputs[0]);
    auto tmp_size = static_cast<SizeType>(taskData->inputs_count[0]);
    data_.input_.assign(tmp_data, tmp_data + tmp_size);
  }
  return true;
}

template <std::copyable DataType, std::unsigned_integral SizeType>
bool RingTopology<DataType, SizeType>::run() {
  internal_order_test();

  // count of processes should be more than 1, but temporary check
  if (world.size() == 1) {
    data_.order_.push_back(0);
    return true;
  }
  auto rank = world.rank();
  int next = (rank == world.size() - 1) ? 0 : rank + 1;
  int prev = (rank == 0) ? world.size() - 1 : rank - 1;

  if (rank == 0) {
    // clear for correct multiply run
    data_.order_.clear();
    world.send(next, Data, data_);
    world.recv(prev, Data, data_);

    data_.order_.emplace_back(rank);
  } else {
    world.recv(prev, Data, data_);
    data_.order_.emplace_back(rank);
    world.send(next, Data, data_);
  }
  return true;
}

template <std::copyable DataType, std::unsigned_integral SizeType>
bool RingTopology<DataType, SizeType>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(data_.input_.begin(), data_.input_.end(), reinterpret_cast<DataType*>(taskData->outputs[0]));
    std::copy(data_.order_.begin(), data_.order_.end(), reinterpret_cast<int*>(taskData->outputs[1]));
  }
  return true;
}

template <std::copyable DataType, std::unsigned_integral SizeType>
std::vector<int> RingTopology<DataType, SizeType>::true_order(int num_processes) {
  std::vector<int> true_order(num_processes);
  std::iota(true_order.begin(), --true_order.end(), 1);
  true_order[num_processes - 1] = 0;
  return true_order;
}

}  // namespace khasanyanov_k_ring_topology_mpi

#endif