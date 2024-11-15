#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kapustin_i_max_cols/include/avg_mpi.hpp"
namespace kapustin_i_max_column_task_mpi {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    int val = gen() % 200000 - 100000;
    vec[i] = val;
  }
  return vec;
}
}  // namespace kapustin_i_max_column_task_mpi
TEST(kapustin_i_max_column_task_mpi, M_5x5_test) {
  boost::mpi::communicator world;
  int cols = 5;
  int rows = 5;
  std::vector<int> matrix;
  std::vector<int> result_parallel(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_vector = cols * rows;
    matrix = kapustin_i_max_column_task_mpi::getRandomVector(size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(result_parallel.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
  }
  kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> res_sequential(cols, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(res_sequential.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);

    // Create Task
    kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(res_sequential, result_parallel);
  }
}
TEST(kapustin_i_max_column_task_mpi, M_1x10_test) {
  boost::mpi::communicator world;
  int cols = 1;
  int rows = 10;
  std::vector<int> matrix;
  std::vector<int> result_parallel(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_vector = cols * rows;
    matrix = kapustin_i_max_column_task_mpi::getRandomVector(size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(result_parallel.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
  }
  kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> res_sequential(cols, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(res_sequential.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);

    kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(res_sequential, result_parallel);
  }
}
TEST(kapustin_i_max_column_task_mpi, M_10x1_test) {
  boost::mpi::communicator world;
  int cols = 10;
  int rows = 1;
  std::vector<int> matrix;
  std::vector<int> result_parallel(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_vector = cols * rows;
    matrix = kapustin_i_max_column_task_mpi::getRandomVector(size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(result_parallel.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
  }
  kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> res_sequential(cols, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(res_sequential.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);

    kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(res_sequential, result_parallel);
  }
}
TEST(kapustin_i_max_column_task_mpi, M_100x100_test) {
  boost::mpi::communicator world;
  int cols = 100;
  int rows = 100;
  std::vector<int> matrix;
  std::vector<int> result_parallel(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_vector = cols * rows;
    matrix = kapustin_i_max_column_task_mpi::getRandomVector(size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_parallel.data()));
    taskDataPar->outputs_count.emplace_back(result_parallel.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
  }
  kapustin_i_max_column_task_mpi::MaxColumnTaskParallelMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> res_sequential(cols, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(res_sequential.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);

    kapustin_i_max_column_task_mpi::MaxColumnTaskSequentialMPI testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(res_sequential, result_parallel);
  }
}