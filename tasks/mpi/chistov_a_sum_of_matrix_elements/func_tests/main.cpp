#include <gtest/gtest.h>

#include "mpi/chistov_a_sum_of_matrix_elements/include/ops_mpi.hpp"

 TEST(chistov_a_sum_of_matrix_elements, test_wrong_validation_parallell) {
   boost::mpi::communicator world;
   std::vector<int> global_matrix;
   std::vector<int32_t> global_sum(2, 0);
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     const int n = 3;
     const int m = 4;
     global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
     taskDataPar->inputs_count.emplace_back(global_matrix.size());
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
     taskDataPar->outputs_count.emplace_back(global_sum.size());

     chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);

     ASSERT_EQ(TestMPITaskParallel.validation(), false);
   }
 }

 //TEST(chistov_a_sum_of_matrix_elements, test_int_sum_parallell) {
 //  boost::mpi::communicator world;

 //  std::vector<int> global_matrix;
 //  std::vector<int32_t> global_sum(1, 0);
 //  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

 //  const int n = 3;
 //  const int m = 4;

 //  if (world.rank() == 0) {
 //    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);

 //    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataPar->inputs_count.emplace_back(global_matrix.size());
 //    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
 //    taskDataPar->outputs_count.emplace_back(global_sum.size());
 //  }

 //  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> testMPITaskParallel(taskDataPar, n, m);
 //  ASSERT_TRUE(testMPITaskParallel.validation());
 //  ASSERT_TRUE(testMPITaskParallel.pre_processing());
 //  ASSERT_TRUE(testMPITaskParallel.run());
 //  ASSERT_TRUE(testMPITaskParallel.post_processing());

 //  if (world.rank() == 0) {
 //    std::vector<int32_t> reference_sum(1, 0);

 //    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
 //    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
 //    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
 //    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

 //    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq, n, m);
 //    ASSERT_TRUE(testMpiTaskSequential.validation());
 //    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
 //    ASSERT_TRUE(testMpiTaskSequential.run());
 //    ASSERT_TRUE(testMpiTaskSequential.post_processing());

 //    ASSERT_EQ(reference_sum[0], global_sum[0]);
 //  }
 //}

 //TEST(chistov_a_sum_of_matrix_elements, test_double_sum_parallell) {
 //  boost::mpi::communicator world;

 //  std::vector<double> global_matrix;
 //  std::vector<double> global_sum(1, 0.0);
 //  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
 //  const int n = 3;
 //  const int m = 4;

 //  if (world.rank() == 0) {
 //    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<double>(n, m);

 //    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataPar->inputs_count.emplace_back(global_matrix.size());
 //    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
 //    taskDataPar->outputs_count.emplace_back(global_sum.size());
 //  }
 //  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<double> testMPITaskParallel(taskDataPar, n, m);
 //  ASSERT_TRUE(testMPITaskParallel.validation());
 //  ASSERT_TRUE(testMPITaskParallel.pre_processing());
 //  ASSERT_TRUE(testMPITaskParallel.run());
 //  ASSERT_TRUE(testMPITaskParallel.post_processing());
 //  if (world.rank() == 0) {
 //    std::vector<double> reference_sum(1, 0.0);
 //    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
 //    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

 //    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
 //    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

 //    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq, n, m);
 //    ASSERT_TRUE(testMpiTaskSequential.validation());
 //    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
 //    ASSERT_TRUE(testMpiTaskSequential.run());
 //    ASSERT_TRUE(testMpiTaskSequential.post_processing());

 //    ASSERT_NEAR(reference_sum[0], global_sum[0], 1e-6);
 //  }
 //}

 //TEST(chistov_a_sum_of_matrix_elements, test_with_empty_matrix_parallell) {
 //  boost::mpi::communicator world;
 //  std::vector<int> global_matrix;
 //  std::vector<int32_t> global_sum(1, 0);

 //  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

 //  if (world.rank() == 0) {
 //    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataPar->inputs_count.emplace_back(global_matrix.size());
 //    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
 //    taskDataPar->outputs_count.emplace_back(global_sum.size());
 //  }

 //  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, 0, 0);
 //  ASSERT_EQ(TestMPITaskParallel.validation(), true);
 //  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
 //  ASSERT_EQ(TestMPITaskParallel.run(), true);
 //  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

 //  if (world.rank() == 0) {
 //    ASSERT_EQ(global_sum[0], 0);
 //  }
 //}

 //TEST(chistov_a_sum_of_matrix_elements, test_with_large_matrix_parallell) {
 //  boost::mpi::communicator world;
 //  std::vector<int> global_matrix;
 //  std::vector<int32_t> global_sum(1, 0);
 //  const int n = 1000;
 //  const int m = 1000;

 //  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

 //  if (world.rank() == 0) {
 //    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
 //    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataPar->inputs_count.emplace_back(global_matrix.size());
 //    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
 //    taskDataPar->outputs_count.emplace_back(global_sum.size());
 //  }

 //  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);
 //  ASSERT_EQ(TestMPITaskParallel.validation(), true);
 //  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
 //  ASSERT_EQ(TestMPITaskParallel.run(), true);
 //  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

 //  if (world.rank() == 0) {
 //    std::vector<int32_t> reference_sum(1, 0);
 //    for (int val : global_matrix) {
 //      reference_sum[0] += val;
 //    }
 //    ASSERT_EQ(reference_sum[0], global_sum[0]);
 //  }
 //}

 //TEST(chistov_a_sum_of_matrix_elements, short_and_thick_test_parallell) {
 //  boost::mpi::communicator world;
 //  std::vector<int> global_matrix;
 //  std::vector<int32_t> global_sum(1, 0);

 //  const int n = 1000000;
 //  const int m = 1;
 //  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

 //  if (world.rank() == 0) {
 //    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
 //    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataPar->inputs_count.emplace_back(global_matrix.size());
 //    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
 //    taskDataPar->outputs_count.emplace_back(global_sum.size());
 //  }

 //  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);
 //  ASSERT_EQ(TestMPITaskParallel.validation(), true);
 //  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
 //  ASSERT_EQ(TestMPITaskParallel.run(), true);
 //  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

 //  if (world.rank() == 0) {
 //    std::vector<int32_t> reference_sum(1, 0);
 //    for (int val : global_matrix) {
 //      reference_sum[0] += val;
 //    }
 //    ASSERT_EQ(reference_sum[0], global_sum[0]);
 //  }
 //}

 //TEST(chistov_a_sum_of_matrix_elements, long_and_thin_test_parallell) {
 //  boost::mpi::communicator world;
 //  std::vector<int> global_matrix;
 //  std::vector<int32_t> global_sum(1, 0);
 //  const int n = 1;
 //  const int m = 100000;

 //  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

 //  if (world.rank() == 0) {
 //    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
 //    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
 //    taskDataPar->inputs_count.emplace_back(global_matrix.size());
 //    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
 //    taskDataPar->outputs_count.emplace_back(global_sum.size());
 //  }

 //  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);
 //  ASSERT_EQ(TestMPITaskParallel.validation(), true);
 //  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
 //  ASSERT_EQ(TestMPITaskParallel.run(), true);
 //  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

 //  if (world.rank() == 0) {
 //    std::vector<int32_t> reference_sum(1, 0);
 //    for (int val : global_matrix) {
 //      reference_sum[0] += val;
 //    }
 //    ASSERT_EQ(reference_sum[0], global_sum[0]);
 //  }
 //}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
