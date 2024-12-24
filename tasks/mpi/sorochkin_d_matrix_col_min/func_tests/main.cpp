#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"
#include "core/task/include/task.hpp"

static void mcm_test(std::vector<int> &&in, uint32_t rows, uint32_t cols) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> out;

  // Create TaskData
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out.resize(cols);
    // in
    taskDataPar->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataPar->inputs_count = {rows, cols};
    // out
    taskDataPar->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataPar->outputs_count = {cols};
  }

  // Create Task
  sorochkin_d_matrix_col_min_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref(cols);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>(*taskDataPar);
    taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(ref.data())};

    sorochkin_d_matrix_col_min_mpi::TestTaskSequential testTaskSeq(taskDataSeq);
    ASSERT_EQ(testTaskSeq.validation(), true);
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    ASSERT_EQ(out, ref);
  }
}

static void mcm_random_test(size_t rows, size_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(rows * cols);
  for (size_t i = 0; i < rows * cols; i++) {
    vec[i] = -50 + gen();
  }

  mcm_test(std::move(vec), rows, cols);
}

TEST(sorochkin_d_matrix_col_min_mpi, Test_1_1) { mcm_random_test(1, 1); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_3_3) { mcm_random_test(3, 3); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_5_5) { mcm_random_test(5, 5); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_5_7) { mcm_random_test(5, 7); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_7_5) { mcm_random_test(7, 5); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_7_7) { mcm_random_test(7, 7); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_13_13) { mcm_random_test(13, 13); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_17_17) { mcm_random_test(17, 17); }
TEST(sorochkin_d_matrix_col_min_mpi, Test_19_19) { mcm_random_test(19, 19); }

TEST(sorochkin_d_matrix_col_min_seq, Test_9_16) {
  // clang-format off
  mcm_test({
    11, 12, 13, 14, 15, 16, 17, 18, 19, 110, 111, 112, 113, 114, 115, 116,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 410, 411, 412, 413, 414, 415, 416,
     1,  2 , 3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 510, 511, 512, 513, 514, 515, 516,
    61, 62, 63, 64, -65, 66, 67, 68, 69, 610, 611, 612, 613, 614, 615, 616,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 710, 711, 712, 713, 714, 715, 716,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 810, 811, 812, 813, 814, 815, 816,
  },9,16);
  // clang-format on
}