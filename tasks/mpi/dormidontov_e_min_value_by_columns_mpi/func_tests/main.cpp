#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/dormidontov_e_min_value_by_columns_mpi/include/ops_mpi.hpp"

TEST(dormidontov_e_min_value_by_columns_mpi, Test_just_test_if_it_finally_works) {
  boost::mpi::communicator world;

  int rs = 7;
  int cs = 7;

  std::vector<int> matrix(cs * rs);
  for (int i = 0; i < cs; i++) {
    for (int j = 0; j < rs; j++) {
      matrix[i * cs + j] = rand() % 1000;
    };
  };
  std::vector<int> res_out_paral(cs, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs);
    taskDataPar->inputs_count.emplace_back(cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs);
    taskDataSeq->inputs_count.emplace_back(cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_min_value_by_columns_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}

TEST(dormidontov_e_min_value_by_columns_mpi, Test_just_test_if_it_finally_works2) {
  boost::mpi::communicator world;

  int rs = 2;
  int cs = 2;

  std::vector<int> matrix(cs * rs);
  for (int i = 0; i < cs; i++) {
    for (int j = 0; j < rs; j++) {
      matrix[i * cs + j] = i * cs + j;
    };
  };
  std::vector<int> res_out_paral(cs);
  res_out_paral[0] = 1;
  res_out_paral[1] = 2;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs);
    taskDataPar->inputs_count.emplace_back(cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> res_out_seq(cs, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rs);
    taskDataSeq->inputs_count.emplace_back(cs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_out_seq.size());
    dormidontov_e_min_value_by_columns_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_out_paral, res_out_seq);
  }
}
