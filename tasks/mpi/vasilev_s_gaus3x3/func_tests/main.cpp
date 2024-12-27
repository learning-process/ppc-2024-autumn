#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/vasilev_s_gaus3x3/include/ops_mpi.hpp"

namespace vasilev_s_gaus3x3_mpi {

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

void run_test(int num_rows, int num_cols) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = vasilev_s_gaus3x3_mpi::getRandomMatrix(num_rows, num_cols);

    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasilev_s_gaus3x3_mpi::Gaus3x3ParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<vasilev_s_gaus3x3_mpi::Gaus3x3SequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}

void validation_test(int num_rows, int num_cols) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = vasilev_s_gaus3x3_mpi::getRandomMatrix(num_rows, num_cols);

    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasilev_s_gaus3x3_mpi::Gaus3x3ParallelMPI>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

}  // namespace vasilev_s_gaus3x3_mpi

TEST(vasilev_s_gaus3x3_mpi, validation_zero_zero) { vasilev_s_gaus3x3_mpi::validation_test(0, 0); }

TEST(vasilev_s_gaus3x3_mpi, validation_one_one) { vasilev_s_gaus3x3_mpi::validation_test(1, 1); }

TEST(vasilev_s_gaus3x3_mpi, validation_one_two) { vasilev_s_gaus3x3_mpi::validation_test(1, 2); }

TEST(vasilev_s_gaus3x3_mpi, validation_two_one) { vasilev_s_gaus3x3_mpi::validation_test(2, 1); }

TEST(vasilev_s_gaus3x3_mpi, validation_two_two) { vasilev_s_gaus3x3_mpi::validation_test(2, 2); }

TEST(vasilev_s_gaus3x3_mpi, validation_two_three) { vasilev_s_gaus3x3_mpi::validation_test(2, 3); }

TEST(vasilev_s_gaus3x3_mpi, validation_three_two) { vasilev_s_gaus3x3_mpi::validation_test(3, 2); }

TEST(vasilev_s_gaus3x3_mpi, three_three) { vasilev_s_gaus3x3_mpi::run_test(3, 3); }

TEST(vasilev_s_gaus3x3_mpi, three_four) { vasilev_s_gaus3x3_mpi::run_test(3, 4); }

TEST(vasilev_s_gaus3x3_mpi, four_three) { vasilev_s_gaus3x3_mpi::run_test(4, 3); }

TEST(vasilev_s_gaus3x3_mpi, five_five) { vasilev_s_gaus3x3_mpi::run_test(5, 5); }

TEST(vasilev_s_gaus3x3_mpi, six_three) { vasilev_s_gaus3x3_mpi::run_test(6, 3); }

TEST(vasilev_s_gaus3x3_mpi, three_six) { vasilev_s_gaus3x3_mpi::run_test(3, 6); }

TEST(vasilev_s_gaus3x3_mpi, seven_five) { vasilev_s_gaus3x3_mpi::run_test(7, 5); }

TEST(vasilev_s_gaus3x3_mpi, five_seven) { vasilev_s_gaus3x3_mpi::run_test(5, 7); }

TEST(vasilev_s_gaus3x3_mpi, eight_eight) { vasilev_s_gaus3x3_mpi::run_test(8, 8); }

TEST(vasilev_s_gaus3x3_mpi, ten_three) { vasilev_s_gaus3x3_mpi::run_test(10, 3); }

TEST(vasilev_s_gaus3x3_mpi, three_ten) { vasilev_s_gaus3x3_mpi::run_test(3, 10); }

TEST(vasilev_s_gaus3x3_mpi, twelve_six) { vasilev_s_gaus3x3_mpi::run_test(12, 6); }

TEST(vasilev_s_gaus3x3_mpi, six_twelve) { vasilev_s_gaus3x3_mpi::run_test(6, 12); }

TEST(vasilev_s_gaus3x3_mpi, fifteen_three) { vasilev_s_gaus3x3_mpi::run_test(15, 3); }

TEST(vasilev_s_gaus3x3_mpi, three_fifteen) { vasilev_s_gaus3x3_mpi::run_test(3, 15); }

TEST(vasilev_s_gaus3x3_mpi, nine_six) { vasilev_s_gaus3x3_mpi::run_test(9, 6); }

TEST(vasilev_s_gaus3x3_mpi, six_nine) { vasilev_s_gaus3x3_mpi::run_test(6, 9); }

TEST(vasilev_s_gaus3x3_mpi, fourteen_fourteen) { vasilev_s_gaus3x3_mpi::run_test(14, 14); }

TEST(vasilev_s_gaus3x3_mpi, eleven_nine) { vasilev_s_gaus3x3_mpi::run_test(11, 9); }

TEST(vasilev_s_gaus3x3_mpi, nine_eleven) { vasilev_s_gaus3x3_mpi::run_test(9, 11); }
