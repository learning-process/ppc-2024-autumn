#ifndef _CIRCLE_TOPOLOGY_HPP_
#define _CIRCLE_TOPOLOGY_HPP_

#include <concepts>
#include <random>
#include <vector>

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

template <Copyable DataType>
class CircleTopology : ppc::core::Task {
 public:
  explicit CircleTopology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};

}  // namespace khasanyanov_k_circle_topology_mpi

#endif