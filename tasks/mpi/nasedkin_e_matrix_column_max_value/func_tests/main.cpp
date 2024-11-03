#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

TEST(MatrixColumnMaxTest, SequentialMaxTest) {
  int size = 100;
  std::vector<int> vec = nasedkin_e_matrix_column_max_value_mpi::getRandomVector(size);
  int expected_max = *std::max_element(vec.begin(), vec.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());
  std::vector<int> output(1, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  auto seqTask =
      std::make_shared<nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskSequential>(taskDataSeq, "max");
  ASSERT_TRUE(seqTask->validation());
  seqTask->pre_processing();
  seqTask->run();
  seqTask->post_processing();

  ASSERT_EQ(output[0], expected_max);
}

TEST(MatrixColumnMaxTest, ParallelMaxTest) {
  boost::mpi::communicator world;
  int size = 100;
  std::vector<int> vec;
  int expected_max = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    vec = nasedkin_e_matrix_column_max_value_mpi::getRandomVector(size);
    expected_max = *std::max_element(vec.begin(), vec.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());
    std::vector<int> output(1, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  auto parTask =
      std::make_shared<nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskParallel>(taskDataPar, "max");
  ASSERT_TRUE(parTask->validation());
  parTask->pre_processing();
  parTask->run();
  parTask->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(reinterpret_cast<int*>(taskDataPar->outputs[0])[0], expected_max);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
