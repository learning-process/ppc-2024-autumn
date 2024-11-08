#ifndef _TESTS_HPP_
#define _TESTS_HPP_

#include <random>

#include "ring_topology.hpp"

namespace khasanyanov_k_ring_topology_mpi {

#define RUN_TASK(task)            \
  ASSERT_TRUE(task.validation()); \
  task.pre_processing();          \
  task.run();                     \
  task.post_processing();

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

template <Copyable DataType>
std::shared_ptr<ppc::core::TaskData> create_task_data(std::vector<DataType>& in, std::vector<int>& out) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(in.size()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(in.size()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));
  return taskData;
}

}  // namespace khasanyanov_k_ring_topology_mpi

#endif