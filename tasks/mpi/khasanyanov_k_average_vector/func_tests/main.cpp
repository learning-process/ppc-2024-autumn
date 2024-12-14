#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "../include/avg_mpi.hpp"
#include "core/task/include/task.hpp"
#include "gtest/gtest.h"

//=========================================sequence=========================================

#define FUNC_SEQ_TEST(InType, OutType, Size, Value)                                                   \
                                                                                                      \
  TEST(khasanyanov_k_average_vector_seq, test_seq_##InType##_##Size) {                                \
    std::vector<InType> in(Size, static_cast<InType>(Value));                                         \
    std::vector<OutType> out(1, 0.0);                                                                 \
    std::shared_ptr<ppc::core::TaskData> taskData =                                                   \
        khasanyanov_k_average_vector_mpi::create_task_data<InType, OutType>(in, out);                 \
    khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<InType, OutType> testTask(taskData); \
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
  std::vector<double> in = khasanyanov_k_average_vector_mpi::get_random_vector<double>(15);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData =
      khasanyanov_k_average_vector_mpi::create_task_data<double, double>(in, out);

  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<double, double> testTask(taskData);
  RUN_TASK(testTask);

  double expect_res = std::accumulate(in.begin(), in.end(), 0.0, std::plus()) / in.size();
  EXPECT_NEAR(out[0], expect_res, 1e-5);
}

//=========================================parallel=========================================

namespace mpi = boost::mpi;

TEST(khasanyanov_k_average_vector_seq, test_displacement) {
  auto displacement = khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<double, double>::displacement(18, 4);
  auto sizes = displacement.first;
  auto displacements = displacement.second;
  std::vector<int> pattern_sizes{5, 5, 4, 4};
  std::vector<int> pattern_displacements{0, 5, 10, 14};
  EXPECT_EQ(sizes, pattern_sizes);
  EXPECT_EQ(displacements, pattern_displacements);
}

TEST(khasanyanov_k_average_vector_mpi, test_wrong_input) {
  mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = khasanyanov_k_average_vector_mpi::create_task_data<double, double>(in, out);
  }
  khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<double, double> testTask(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testTask.validation());
  }
}

#define FUNC_MPI_TEST(InType, OutType, Size)                                                               \
  TEST(khasanyanov_k_average_vector_mpi, test_mpi_##InType##_##Size) {                                     \
    mpi::communicator world;                                                                               \
    std::vector<InType> in = khasanyanov_k_average_vector_mpi::get_random_vector<InType>(Size);            \
    std::vector<OutType> out(1, 0.0);                                                                      \
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();               \
    if (world.rank() == 0) {                                                                               \
      taskData = khasanyanov_k_average_vector_mpi::create_task_data<InType, OutType>(in, out);             \
    }                                                                                                      \
    khasanyanov_k_average_vector_mpi::AvgVectorMPITaskParallel<InType, OutType> testTask(taskData);        \
    RUN_TASK(testTask);                                                                                    \
    if (world.rank() == 0) {                                                                               \
      std::vector<OutType> seq_out(1, 0.0);                                                                \
      std::shared_ptr<ppc::core::TaskData> taskDataSeq =                                                   \
          khasanyanov_k_average_vector_mpi::create_task_data<InType, OutType>(in, seq_out);                \
                                                                                                           \
      khasanyanov_k_average_vector_mpi::AvgVectorMPITaskSequential<InType, OutType> testMpiTaskSequential( \
          taskDataSeq);                                                                                    \
                                                                                                           \
      RUN_TASK(testMpiTaskSequential);                                                                     \
      EXPECT_NEAR(seq_out[0], out[0], 1e-5);                                                               \
    }                                                                                                      \
  }

#define RUN_FUNC_MPI_TESTS(Size)        \
  FUNC_MPI_TEST(int8_t, double, Size)   \
  FUNC_MPI_TEST(int16_t, double, Size)  \
  FUNC_MPI_TEST(int32_t, double, Size)  \
  FUNC_MPI_TEST(int64_t, double, Size)  \
  FUNC_MPI_TEST(uint8_t, double, Size)  \
  FUNC_MPI_TEST(uint16_t, double, Size) \
  FUNC_MPI_TEST(uint32_t, double, Size) \
  FUNC_MPI_TEST(uint64_t, double, Size) \
  FUNC_MPI_TEST(double, double, Size)   \
  FUNC_MPI_TEST(float, double, Size)

#define RUN_FUNC_TESTS(Size, Value) \
  RUN_FUNC_SEQ_TESTS(Size, Value)   \
  RUN_FUNC_MPI_TESTS(Size)

#define RUN_ALL_FUNC_TESTS() \
  RUN_FUNC_TESTS(1234, 7.7)  \
  RUN_FUNC_TESTS(2000, 10)   \
  RUN_FUNC_TESTS(9, 77)      \
  RUN_FUNC_TESTS(3011, 111)  \
  RUN_FUNC_TESTS(2, 23)

//=======run=============
RUN_ALL_FUNC_TESTS()