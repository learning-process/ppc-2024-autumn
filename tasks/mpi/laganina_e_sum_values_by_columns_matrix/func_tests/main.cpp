#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/laganina_e_sum_values_by_columns_matrix/include/ops_mpi.hpp"

TEST(laganina_e_sum_values_by_columns_matrix_mpi, Test_2_2_matrix) {
   boost::mpi::communicator world;
   std::vector<int> in = { 1,2,1,2 };
   int n = 2;
   int m = 2;
   std::vector<int> empty_par(n, 0);
   std::vector<int> out = { 2,4 };
   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
    taskDataPar->outputs_count.emplace_back(empty_par.size());
   }
   laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
   ASSERT_EQ(testMpiTaskParallel.validation(), true);
   testMpiTaskParallel.pre_processing();
   testMpiTaskParallel.run();
   testMpiTaskParallel.post_processing();
   if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());
    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(empty_par, empty_seq);
   }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, Test_500_300_matrix) {
  boost::mpi::communicator world;
  int n = 300;
  int m = 500;
  // Create data
  std::vector<int> in = laganina_e_sum_values_by_columns_matrix_mpi::getRandomVector(n * m);
  std::vector<int> empty_par(n, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
    taskDataPar->outputs_count.emplace_back(empty_par.size());
   }
   laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
   ASSERT_EQ(testMpiTaskParallel.validation(), true);
   testMpiTaskParallel.pre_processing();
   testMpiTaskParallel.run();
   testMpiTaskParallel.post_processing();
   if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());
    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(empty_par, empty_seq);
   }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, partest1) {
 boost::mpi::communicator world;
 int  n = 2;
 int m = 2;
 // Create data
 std::vector<int> in = { 1, 2, 1, 2 };
 std::vector<int> empty_par(n, 0);
 std::vector<int> out = { 2, 4 };
 // Create TaskData
 std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
   taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
   taskDataPar->inputs_count.emplace_back(in.size());
   taskDataPar->inputs_count.emplace_back(m);
   taskDataPar->inputs_count.emplace_back(n);
   taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
   taskDataPar->outputs_count.emplace_back(empty_par.size());
  }
   laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
   ASSERT_EQ(testMpiTaskParallel.validation(), true);
   testMpiTaskParallel.pre_processing();
   testMpiTaskParallel.run();
   testMpiTaskParallel.post_processing();
   if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());
    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(empty_par, empty_seq);
   }
}
TEST(laganina_e_sum_values_by_columns_matrix_mpi, partest2) {
 boost::mpi::communicator world;
 int  n = 5000;
 int m = 3000;
 // Create data
 std::vector<int> in(m * n, 1);
 std::vector<int> empty_par(n, 0);
 std::vector<int> out(n, m);
 // Create TaskData
 std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
 if (world.rank() == 0) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs_count.emplace_back(m);
  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
  taskDataPar->outputs_count.emplace_back(empty_par.size());
 }
 laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
 ASSERT_EQ(testMpiTaskParallel.validation(), true);
 testMpiTaskParallel.pre_processing();
 testMpiTaskParallel.run();
 testMpiTaskParallel.post_processing();
if (world.rank() == 0) {
 // Create data
 std::vector<int> empty_seq(n, 0);
 // Create TaskData
 std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
 taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
 taskDataSeq->inputs_count.emplace_back(in.size());
 taskDataSeq->inputs_count.emplace_back(m);
 taskDataSeq->inputs_count.emplace_back(n);
 taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
 taskDataSeq->outputs_count.emplace_back(empty_seq.size());
 // Create Task
 laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
 ASSERT_EQ(testMpiTaskSequential.validation(), true);
 testMpiTaskSequential.pre_processing();
 testMpiTaskSequential.run();
 testMpiTaskSequential.post_processing();
 ASSERT_EQ(empty_par, empty_seq);
 }
}
TEST(laganina_e_sum_values_by_columns_matrix_mpi, partest3) {
 boost::mpi::communicator world;
 int  n = 3000;
 int m = 5000;
 // Create data
 std::vector<int> in(m * n, 1);
 std::vector<int> empty_par(n, 0);
 std::vector<int> out(n, m);
 // Create TaskData
 std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
 if (world.rank() == 0) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs_count.emplace_back(m);
  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
  taskDataPar->outputs_count.emplace_back(empty_par.size());
 }
 laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
 ASSERT_EQ(testMpiTaskParallel.validation(), true);
 testMpiTaskParallel.pre_processing();
 testMpiTaskParallel.run();
 testMpiTaskParallel.post_processing();
 if (world.rank() == 0) {
 // Create data
  std::vector<int> empty_seq(n, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
  taskDataSeq->outputs_count.emplace_back(empty_seq.size());
  // Create Task
  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_EQ(empty_par, empty_seq);
 }
}