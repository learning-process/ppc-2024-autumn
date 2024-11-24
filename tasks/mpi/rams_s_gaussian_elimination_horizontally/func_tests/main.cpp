#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/rams_s_gaussian_elimination_horizontally/include/ops_mpi.hpp"

void rams_s_gaussian_elimination_horizontally_mpi_run_test(std::vector<double> &&in, size_t variables_count) {
  boost::mpi::communicator world;
  std::vector<double> out(variables_count, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_out(variables_count, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < variables_count; i++) {
      if (std::isnan(reference_out[i])) {
        ASSERT_TRUE(std::isnan(out[i]));
      } else {
        ASSERT_DOUBLE_EQ(reference_out[i], out[i]);
      }
    }
  }
}
#define TEST_IT(case_name, ...)                                         \
  TEST(rams_s_gaussian_elimination_horizontally_mpi, case_name) {       \
    rams_s_gaussian_elimination_horizontally_mpi_run_test(__VA_ARGS__); \
  }

// clang-format off
TEST_IT(0, {}, 0)

TEST_IT(1, {
     0,  0,  3,  6,
     7,  0,  0,  0,
     0,  4,  0, -2,
  }, 3)

TEST_IT(2, {
     0,  0,  3,  6,
     0,  2,  0, -4,
     7,  0,  0,  1,
     0,  0,  6, 12,
  }, 3)

TEST_IT(2_1, {
     0,  0,  3,  6,
     7,  0,  0,  1,
     0,  2,  0, -4,
     0,  0,  6, 12,
  }, 3)

TEST_IT(3, {
     0,  1, -2,  0,
    -1, -2,  1, -1,
     2,  3,  0,  2
  }, 3)

TEST_IT(4, {
     0,  1, -2,  0,
     0,  2, -4,  0,
     2,  0,  0, -4
  }, 3)

TEST_IT(5, {
     7,  0,  2, 11,
     0,  0,  4,  8,
     0,  0,  2,  4
  }, 3)

TEST_IT(6, {
     7,  1,  2, 12,
     0,  0,  4,  8,
     0,  0,  2,  4
  }, 3)

TEST_IT(7, {
     7,  1,  2, 12,
     0,  5,  0,  5,
     0,  0,  4,  8,
     0,  0,  2,  4
  }, 3)

TEST_IT(8, {
     7,  1,  2, 12,
  }, 3)

TEST_IT(9, {
     1,  1,  1,  1,  1,  1,  1, -7,
     1,  2,  2,  2,  2,  2,  2,-13,
     1,  2,  3,  3,  3,  3,  3,-17,
     1,  2,  3,  4,  4,  4,  4,-22,
     1,  2,  3,  4,  5,  5,  5,-25,
     1,  2,  3,  4,  5,  6,  6,-27,
     1,  2,  3,  4,  5,  6,  7,-28,
  }, 7)
// clang-format on

void rams_s_gaussian_elimination_horizontally_mpi_test_random_matrix(int variables_count) {
  std::vector<double> in(variables_count * (variables_count + 1));
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int row = 0; row < variables_count; row++) {
    for (int col = 0; col < variables_count + 1; col++) {
      int k = 0;
      while ((k = gen()) == 0);
      in[row * (variables_count + 1) + col] = k;
    }
  }
  rams_s_gaussian_elimination_horizontally_mpi_run_test(std::move(in), variables_count);
}

#define TEST_IT_RANDOM(case_name, variables_count)                                    \
  TEST(rams_s_gaussian_elimination_horizontally_mpi, random_##case_name) {            \
    rams_s_gaussian_elimination_horizontally_mpi_test_random_matrix(variables_count); \
  }

TEST_IT_RANDOM(1, 3)
TEST_IT_RANDOM(2, 13)
TEST_IT_RANDOM(3, 16)
TEST_IT_RANDOM(4, 99)
TEST_IT_RANDOM(5, 56)
