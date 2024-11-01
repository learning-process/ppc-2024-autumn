// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/shlyakov_m_min_value_of_row/include/ops_mpi.hpp"

TEST(shlyakov_m_min_value_of_row_mpi, test_validation) {
  boost::mpi::communicator world;
  const int sz_row = 100;
  const int sz_col = 100;

  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min(sz_row, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    main_matr =
        shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataPar->inputs_count = {sz_row, sz_col};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
}

TEST(shlyakov_m_min_value_of_row_mpi, test_pre_processing) {
  boost::mpi::communicator world;
  const int sz_row = 100;
  const int sz_col = 100;

  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min(sz_row, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    main_matr =
        shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataPar->inputs_count = {sz_row, sz_col};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
}

TEST(shlyakov_m_min_value_of_row_mpi, test_run) {
  boost::mpi::communicator world;
  const int sz_row = 100;
  const int sz_col = 100;

  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min(sz_row, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    main_matr =
        shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataPar->inputs_count = {sz_row, sz_col};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  ASSERT_TRUE(testMpiTaskParallel.run());
}

TEST(shlyakov_m_min_value_of_row_mpi, test_post_processing) {
  boost::mpi::communicator world;
  const int sz_row = 100;
  const int sz_col = 100;

  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min(sz_row, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    main_matr = shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataPar->inputs_count = {sz_row, sz_col};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(shlyakov_m_min_value_of_row_mpi, test_with_square_matr) {
  boost::mpi::communicator world;
  const int sz_row = 100;
  const int sz_col = 100;

  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min(sz_row, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    main_matr =
        shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataPar->inputs_count = {sz_row, sz_col};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_min(sz_row, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataSeq->inputs_count = {sz_row, sz_col};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_min.data()));
    taskDataSeq->outputs_count.emplace_back(seq_min.size());

    shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < sz_row; i++) {
      ASSERT_EQ(main_min[i], INT_MIN);
    }
  }
}

TEST(shlyakov_m_min_value_of_row_mpi, test_with_not_square_matr) {
  boost::mpi::communicator world;
  const int sz_row = 400;
  const int sz_col = 100;

  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min(sz_row, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    main_matr = shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataPar->inputs_count = {sz_row, sz_col};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_min(sz_row, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataSeq->inputs_count = {sz_row, sz_col};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_min.data()));
    taskDataSeq->outputs_count.emplace_back(seq_min.size());

    shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < sz_row; i++) {
      ASSERT_EQ(main_min[i], INT_MIN);
    }
  }
}

TEST(shlyakov_m_min_value_of_row_mpi, test_with_large_matr) {
  boost::mpi::communicator world;
  const int sz_row = 5000;
  const int sz_col = 5000;

  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min(sz_row, INT_MAX);
  
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    main_matr = shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataPar->inputs_count = {sz_row, sz_col};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_min(sz_row, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < main_matr.size(); i++) taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(main_matr[i].data()));

    taskDataSeq->inputs_count = {sz_row, sz_col};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_min.data()));
    taskDataSeq->outputs_count.emplace_back(seq_min.size());

    shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < sz_row; i++) {
      ASSERT_EQ(main_min[i], INT_MIN);
    }
  }
}

TEST(shlyakov_m_min_value_of_row_mpi, test_with_empty_input) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int sz_row = 100;
    const int sz_col = 100;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matrix_rnd =
        shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);

    taskDataSeq->inputs_count.emplace_back(sz_row);
    taskDataSeq->inputs_count.emplace_back(sz_col);

    std::vector<int> v_res(sz_row, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
    taskDataSeq->outputs_count.emplace_back(v_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}