#ifndef _TESTS_HPP_
#define _TESTS_HPP_

#include <gtest/gtest.h>

#include <random>

#include "ring_topology.hpp"

namespace khasanyanov_k_ring_topology_mpi {

#define RUN_TASK(task)              \
  ASSERT_TRUE((task).validation()); \
  (task).pre_processing();          \
  (task).run();                     \
  (task).post_processing();

// not included 'right' border with integers, not included 'left' border always
template <typename T = int>
std::vector<T> generate_random_vector(size_t size = 100, const T& left = static_cast<T>(-1000),
                                      const T& right = static_cast<T>(1000)) {
  std::vector<T> res(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  double frac = (gen() % 100) / 100.0;
  for (size_t i = 0; i < size; i++) {
    res[i] = left + frac + static_cast<T>(gen() % static_cast<int>(right - left));
  }
  return res;
}

template <std::copyable DataType>
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

#ifndef FUNC_TEST
#define FUNC_TEST(DataType, SizeType, Size)                                                                          \
  TEST(khasanyanov_k_ring_topology_tests, ring_topology_test_data_type_##DataType##_size_type_##SizeType##_##Size) { \
    boost::mpi::communicator world;                                                                                  \
    const std::vector<DataType> in_data =                                                                            \
        khasanyanov_k_ring_topology_mpi::generate_random_vector<DataType>(static_cast<SizeType>(Size));              \
    std::vector<DataType> out_data(in_data);                                                                         \
    std::vector<int> order(world.size());                                                                            \
    auto taskData = std::make_shared<ppc::core::TaskData>();                                                         \
    if (world.rank() == 0) {                                                                                         \
      taskData = khasanyanov_k_ring_topology_mpi::create_task_data<DataType>(out_data, order);                       \
    }                                                                                                                \
    khasanyanov_k_ring_topology_mpi::RingTopology<DataType, SizeType> testTask(taskData);                            \
    RUN_TASK(testTask);                                                                                              \
    if (world.rank() == 0) {                                                                                         \
      auto pattern_order =                                                                                           \
          khasanyanov_k_ring_topology_mpi::RingTopology<DataType, SizeType>::true_order(world.size());               \
      ASSERT_EQ(pattern_order, order);                                                                               \
      ASSERT_EQ(in_data, out_data);                                                                                  \
    }                                                                                                                \
  }
#endif

#ifndef RUN_FUNC_TESTS_BY_SIZE_TYPE
#define RUN_FUNC_TESTS_BY_SIZE_TYPE(SizeType, Size) \
  FUNC_TEST(int8_t, SizeType, Size)                 \
  FUNC_TEST(int16_t, SizeType, Size)                \
  FUNC_TEST(int32_t, SizeType, Size)                \
  FUNC_TEST(int64_t, SizeType, Size)                \
  FUNC_TEST(uint8_t, SizeType, Size)                \
  FUNC_TEST(uint16_t, SizeType, Size)               \
  FUNC_TEST(uint32_t, SizeType, Size)               \
  FUNC_TEST(uint64_t, SizeType, Size)               \
  FUNC_TEST(double, SizeType, Size)                 \
  FUNC_TEST(float, SizeType, Size)
#endif

#ifndef RUN_FUNC_TESTS_BY_SIZE
#define RUN_FUNC_TESTS_BY_SIZE(Size)          \
  RUN_FUNC_TESTS_BY_SIZE_TYPE(uint8_t, Size)  \
  RUN_FUNC_TESTS_BY_SIZE_TYPE(uint16_t, Size) \
  RUN_FUNC_TESTS_BY_SIZE_TYPE(uint32_t, Size)
#endif

#ifndef RUN_FUNC_TESTS
#define RUN_FUNC_TESTS()       \
  RUN_FUNC_TESTS_BY_SIZE(1)    \
  RUN_FUNC_TESTS_BY_SIZE(250)  \
  RUN_FUNC_TESTS_BY_SIZE(500)  \
  RUN_FUNC_TESTS_BY_SIZE(1000) \
  RUN_FUNC_TESTS_BY_SIZE(10000)
#endif

}  // namespace khasanyanov_k_ring_topology_mpi

#endif