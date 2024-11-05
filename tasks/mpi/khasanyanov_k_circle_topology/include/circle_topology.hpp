#ifndef _CIRCLE_TOPOLOGY_HPP_
#define _CIRCLE_TOPOLOGY_HPP_

#include <concepts>
#include <cstddef>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace khasanyanov_k_circle_topology_mpi {

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
class CircleTopology : ppc::core::Task {
  static_assert(sizeof(SizeType) > sizeof(std::uint32_t),
                "Size of 'SizeType' greater than std::uint32_t, possible loss of data");

 private:
  boost::mpi::communicator world;
  std::vector<DataType> data_;

 public:
  explicit CircleTopology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
};

template <Copyable DataType, Unsigned SizeType>
bool CircleTopology<DataType, SizeType>::validation() {
  if (world.rank() == 0) {
    return !taskData->inputs.empty() && !taskData->inputs_count.empty() && taskData->inputs_count[0] > 0;
  }
  return true;
}

template <Copyable DataType, Unsigned SizeType>
bool CircleTopology<DataType, SizeType>::pre_processing() {
  if (world.rank() == 0) {
    auto* tmp_data = reinterpret_cast<DataType*>(taskData->inputs[0]);
    auto* tmp_size = reinterpret_cast<SizeType*>(taskData->inputs_count[0]);
    data_.assign(tmp_data, tmp_data + *tmp_size);
  }
  return true;
}

}  // namespace khasanyanov_k_circle_topology_mpi

#endif