#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/chistov_a_sum_of_matrix_elements/include/ops_mpi.hpp"

TEST(chistov_a_sum_of_matrix_elements, test_int_sum_sequential) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    int n = 3;
    int m = 4;
    std::vector<int> global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
    std::vector<int32_t> reference_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq, n, m);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
    ASSERT_EQ(testMpiTaskSequential.run(), true);
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);

    int sum = chistov_a_sum_of_matrix_elements::classic_way(global_matrix, n, m);
    ASSERT_EQ(reference_sum[0], sum);
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_double_sum_sequential) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    int n = 3;
    int m = 4;

    std::vector<double> global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<double>(n, m);

    std::vector<double> reference_sum(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq, n, m);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
    ASSERT_EQ(testMpiTaskSequential.run(), true);
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);

    double sum = chistov_a_sum_of_matrix_elements::classic_way(global_matrix, n, m);

    ASSERT_NEAR(reference_sum[0], sum, 1e-6);
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_sum_with_empty_matrix_sequential) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);
    std::vector<int> empty_matrix;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(empty_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(empty_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());
    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq, 0, 0);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
    ASSERT_EQ(testMpiTaskSequential.run(), true);
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);

    ASSERT_EQ(reference_sum[0], 0);
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_sum_with_single_element_matrix_sequential) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    int n = 1;
    int m = 1;
    std::vector<int> global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
    std::vector<int32_t> reference_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq, n, m);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
    ASSERT_EQ(testMpiTaskSequential.run(), true);
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);

    int sum = chistov_a_sum_of_matrix_elements::classic_way(global_matrix, n, m);
    ASSERT_EQ(reference_sum[0], sum);
  }
}

TEST(chistov_a_sum_of_matrix_elements, throws_when_small_n_or_m_sequential) {
  boost::mpi::communicator world;
  int n, m;
  if (world.rank() == 0) {
    n = 0;
    m = 4;

    EXPECT_THROW({ chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m); }, std::invalid_argument);

    m = 0;
    EXPECT_THROW({ chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(1, m); }, std::invalid_argument);
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_wrong_validation_sequential) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int n, m;
  if (world.rank() == 0) {
    n = 3;
    m = 4;
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataSeq->outputs_count.emplace_back(global_sum.size());
  }
  chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq, n, m);
  ASSERT_EQ(testMpiTaskSequential.validation(), false);
}

TEST(chistov_a_sum_of_matrix_elements, test_wrong_validation_parallell) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int n, m;
  if (world.rank() == 0) {
    n = 3;
    m = 4;
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);

    ASSERT_EQ(TestMPITaskParallel.validation(), false);
  }
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
