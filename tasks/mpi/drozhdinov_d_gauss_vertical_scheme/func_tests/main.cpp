// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
// not example
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/drozhdinov_d_gauss_vertical_scheme/include/ops_mpi.hpp"

TEST(MPIGAUSS, EquationTest) {
  boost::mpi::communicator world;
  int rows = 2;
  int columns = 2;
  std::vector<double> matrix = {1, 0, 0, 1};
  std::vector<double> b = {1, 1};
  std::vector<double> expres_par(rows);
  std::vector<double> res_par = {1, 1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, Size100Test) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<double> matrix = genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> res_par(rows, 1);
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, EmptyTest) {
  boost::mpi::communicator world;
  int rows = 0;
  int columns = 0;
  std::vector<double> matrix = {};
  std::vector<double> b = {};
  std::vector<double> expres_par(rows, 0);
  std::vector<double> res_par = {};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, Size10000Test) {
  boost::mpi::communicator world;
  int rows = 100;
  int columns = 100;
  std::vector<double> matrix = genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> res_par(rows, 1);
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, Size490000Test) {
  boost::mpi::communicator world;
  int rows = 700;
  int columns = 700;
  std::vector<double> matrix = genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> res_par(rows, 1);
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

/*
int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}*/
