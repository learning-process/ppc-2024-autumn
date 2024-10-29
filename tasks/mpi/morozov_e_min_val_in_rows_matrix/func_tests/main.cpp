// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/morozov_e_min_val_in_rows_matrix/include/ops_mpi.hpp"

TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Validation_isFalse) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  for (int i = 0; i < matrix.size(); ++i) 
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataSeq);
  ASSERT_FALSE(testMpiTaskSequential.validation());
  ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Validation_isTrue) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  std::vector<int> res = {1, 2};
  for (int i = 0; i < matrix.size(); ++i) taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(2);
  morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
  ASSERT_TRUE(testMpiTaskParallel.validation());

}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Main) {
  std::vector<std::vector<int>> matrix;
  const int n = 1000;
  const int m = 1000;
  std::vector<int> resSeq(n);
  std::vector<int> resPar(n);
  std::vector<int> res(n);
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  matrix = morozov_e_min_val_in_rows_matrix::getRandomMatrix(n, m);
  res = morozov_e_min_val_in_rows_matrix::minValInRowsMatrix(matrix);
  for (int i = 0; i < matrix.size(); ++i) {
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  }
    
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(resPar.size());

   taskDataSeq->inputs_count.emplace_back(n);
   taskDataSeq->inputs_count.emplace_back(m);
   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
   taskDataSeq->outputs_count.emplace_back(resSeq.size());
  morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  //ASSERT_EQ(v, res2);
  ASSERT_EQ(resPar, resSeq);
  ASSERT_EQ(resSeq, res);
}

