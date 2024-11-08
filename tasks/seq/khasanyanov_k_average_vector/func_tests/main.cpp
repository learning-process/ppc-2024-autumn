#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "../include/avg_seq.hpp"
#include "core/task/include/task.hpp"
#include "gtest/gtest.h"

#define FUNC_SEQ_TEST(InType, OutType, Size, Value)                                                   \
                                                                                                      \
  TEST(khasanyanov_k_average_vector_seq, test_seq_##InType##_##Size) {                                \
    std::vector<InType> in(Size, static_cast<InType>(Value));                                         \
    std::vector<OutType> out(1, 0.0);                                                                 \
    std::shared_ptr<ppc::core::TaskData> taskData =                                                   \
        khasanyanov_k_average_vector_seq::create_task_data<InType, OutType>(in, out);                 \
    khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<InType, OutType> testTask(taskData); \
    RUN_TASK(testTask);                                                                               \
    EXPECT_NEAR(out[0], static_cast<InType>(Value), 1e-5);                                            \
  }

#define RUN_FUNC_SEQ_TESTS(Size, Value)        \
  FUNC_SEQ_TEST(int8_t, double, Size, Value)   \
  FUNC_SEQ_TEST(int16_t, double, Size, Value)  \
  FUNC_SEQ_TEST(int32_t, double, Size, Value)  \
  FUNC_SEQ_TEST(int64_t, double, Size, Value)  \
  FUNC_SEQ_TEST(uint8_t, double, Size, Value)  \
  FUNC_SEQ_TEST(uint16_t, double, Size, Value) \
  FUNC_SEQ_TEST(uint32_t, double, Size, Value) \
  FUNC_SEQ_TEST(uint64_t, double, Size, Value) \
  FUNC_SEQ_TEST(double, double, Size, Value)   \
  FUNC_SEQ_TEST(float, double, Size, Value)

TEST(khasanyanov_k_average_vector_seq, test_random) {
  std::vector<double> in = khasanyanov_k_average_vector_seq::get_random_vector<double>(15);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_seq::create_task_data<double, double>(in, out);

  khasanyanov_k_average_vector_seq::AvgVectorSEQTaskSequential<double, double> testTask(taskData);
  RUN_TASK(testTask);

  double expect_res = std::accumulate(in.begin(), in.end(), 0.0, std::plus()) / in.size();
  EXPECT_NEAR(out[0], expect_res, 1e-5);
}

#define RUN_ALL_FUNC_TESTS()    \
  RUN_FUNC_SEQ_TESTS(1234, 7.7) \
  RUN_FUNC_SEQ_TESTS(2000, 10)  \
  RUN_FUNC_SEQ_TESTS(9, 77)     \
  RUN_FUNC_SEQ_TESTS(3011, 111)

RUN_ALL_FUNC_TESTS()