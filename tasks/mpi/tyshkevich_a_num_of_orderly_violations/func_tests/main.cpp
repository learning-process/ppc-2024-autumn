#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>
#include <vector>

#include "mpi/tyshkevich_a_num_of_orderly_violations/include/ops_mpi.hpp"

std::vector<int> tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

std::string VecToStrTY(std::vector<int> &v) {
  std::ostringstream oss;

  if (!v.empty()) {
    std::copy(v.begin(), v.end() - 1, std::ostream_iterator<int>(oss, ","));
    oss << v.back();
  }
  return oss.str();
}

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ftest, Test_Max_10) {
  int size = 10;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size);
  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(result, local_count) << VecToStrTY(global_vec) << ' ' << size << ' ' << world.size() << std::endl;
  }
}

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ftest, Test_Max_20) {
  int size = 20;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size);
  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(result, local_count) << VecToStrTY(global_vec) << ' ' << size << ' ' << world.size() << std::endl;
  }
}

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ftest, Test_Max_50) {
  int size = 50;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size);
  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(result, local_count) << VecToStrTY(global_vec) << ' ' << size << ' ' << world.size() << std::endl;
  }
}