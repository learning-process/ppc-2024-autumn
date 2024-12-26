#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/dormidontov_e_min_value_by_columns_mpi/include/ops_mpi.hpp"
boost::mpi::communicator world;

inline std::vector<int> generate_random_vector(int cs_temp, int rs_temp) {
  std::vector<int> temp(cs_temp * rs_temp);
  for (int i = 0; i < rs_temp; i++) {
    for (int j = 0; j < cs_temp; j++) {
      if (i == 0) {
        temp[i * rs_temp + j] = 0;
      } else {
        temp[i * rs_temp + j] = (rand() % 1999) - 999;
      }
    }
  }
  return temp;
}

TEST(dormidontov_e_min_value_by_columns_mpi, Test_just_test_if_it_finally_works) {
  int rs = 7;
  int cs = 7;

  std::vector<int> matrix(cs * rs);
  matrix = generate_random_vector(cs, rs);
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
  int rs = 2;
  int cs = 2;

  std::vector<int> matrix(cs * rs);
  matrix = generate_random_vector(cs, rs);
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

TEST(dormidontov_e_min_value_by_columns_mpi, Test_Empty) {
  const int rs = 0;
  const int cs = 0;

  std::vector<int> matrix = {};
  std::vector<int> res_out_paral(cs, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rs);
    taskDataPar->inputs_count.emplace_back(cs);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
    dormidontov_e_min_value_by_columns_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(dormidontov_e_min_value_by_columns_mpi, Test_just_test_if_it_finally_works5) {
  int rs = 2000;
  int cs = 2000;

  std::vector<int> matrix(cs * rs);
  matrix = generate_random_vector(cs, rs);
  std::vector<int> res_out_paral(cs);

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

TEST(dormidontov_e_min_value_by_columns_mpi, Test_just_test_if_it_finally_works6) {
  int rs = 20;
  int cs = 30;

  std::vector<int> matrix(cs * rs);
  matrix = generate_random_vector(cs, rs);
  std::vector<int> res_out_paral(cs);

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