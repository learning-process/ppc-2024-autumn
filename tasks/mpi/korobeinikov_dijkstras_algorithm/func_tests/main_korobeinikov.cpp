// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/korobeinikov_dijkstras_algorithm/include/ops_mpi_korobeinikov.hpp"

namespace korobeinikov_a_test_task_mpi_lab_03 {

std::vector<int> getRandomVector(int sz, int size_cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % size_cols;
  }
  return vec;
}

void getRandomGraph(int sz, int count_edges, std::vector<int> &val, std::vector<int> &col, std::vector<int> &ri) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < count_edges; i++) {
    val[i] = gen() % 10;
    col[i] = gen() % sz;
  }
  ri[0] = 0;
  ri[sz] = count_edges;
  std::vector<int> tmp(sz - 1);
  for (int i = 0; i < sz - 1; i++) {
    tmp[i] = gen() % count_edges;
  }
  std::sort(tmp.begin(), tmp.end());
  for (int i = 1; i < sz; i++) {
    ri[i] = tmp[i - 1];
  }
}

}  // namespace korobeinikov_a_test_task_mpi_lab_03

TEST(korobeinikov_dijkstras_algorithm_mpi, Test_1_const_numbers) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> values = {10, 30, 100, 50, 10, 20, 60};
  std::vector<int> col = {1, 3, 4, 2, 4, 2, 4};
  std::vector<int> RowIndex = {0, 3, 4, 5, 7, 7};

  int size = 5;
  int sv = 0;

  std::vector<int> out(size, 0);
  std::vector<int> right_answer = {0, 10, 50, 30, 60};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataPar->inputs_count.emplace_back(col.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataPar->inputs_count.emplace_back(RowIndex.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    //// Create data
    std::vector<int> out_seq(size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataSeq->inputs_count.emplace_back(col.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataSeq->inputs_count.emplace_back(RowIndex.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    // Create Task
    korobeinikov_a_test_task_mpi_lab_03::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out_seq[i]);
    }
    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out[i]);
    }
  }
}

TEST(korobeinikov_dijkstras_algorithm_mpi, Test_2_random_numbers) {
  boost::mpi::communicator world;
  // Create data
  int size = 5;
  int sv = 0;

  std::vector<int> values(7);
  std::vector<int> col(7);
  std::vector<int> RowIndex(size + 1);

  std::vector<int> out(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    korobeinikov_a_test_task_mpi_lab_03::getRandomGraph(5, 7, values, col, RowIndex);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataPar->inputs_count.emplace_back(col.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataPar->inputs_count.emplace_back(RowIndex.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> out_seq(size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataSeq->inputs_count.emplace_back(col.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataSeq->inputs_count.emplace_back(RowIndex.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    // Create Task
    korobeinikov_a_test_task_mpi_lab_03::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(out[i], out_seq[i]);
    }
  }
}

TEST(korobeinikov_dijkstras_algorithm_mpi, Test_3_empty_graph) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> values;
  std::vector<int> col;
  std::vector<int> RowIndex;

  int size = 5;
  int sv = 0;

  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    values = korobeinikov_a_test_task_mpi_lab_03::getRandomVector(7, size);
    col = korobeinikov_a_test_task_mpi_lab_03::getRandomVector(7, size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataPar->inputs_count.emplace_back(col.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataPar->inputs_count.emplace_back(RowIndex.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(korobeinikov_dijkstras_algorithm_mpi, Test_4_validation_false_sv_greater_size) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> values(7);
  std::vector<int> col(7);
  std::vector<int> RowIndex = {0, 3, 4, 5, 7, 7};

  int size = 10;
  int sv = 0;

  std::vector<int> out(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    values = korobeinikov_a_test_task_mpi_lab_03::getRandomVector(7, size);
    col = korobeinikov_a_test_task_mpi_lab_03::getRandomVector(7, size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataPar->inputs_count.emplace_back(col.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataPar->inputs_count.emplace_back(RowIndex.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}