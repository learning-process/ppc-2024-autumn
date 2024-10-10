#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include "mpi/chistov_a_sum_of_matrix_elements/include/ops_mpi.hpp"

TEST(TestMPITaskSequential, test_sum) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
     int n = 3;
     int m = 4;
    std::vector<int> global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);
    std::vector<int32_t> reference_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, n,m);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
    ASSERT_EQ(testMpiTaskSequential.run(), true);
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);

    int sum = chistov_a_sum_of_matrix_elements::classic_way(global_matrix, n, m);
    ASSERT_EQ(reference_sum[0], sum);  
  }
}

TEST(TestMPITaskSequential, test_sum_with_empty_matrix) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);
    std::vector<int> empty_matrix;  

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(empty_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(empty_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());
    chistov_a_sum_of_matrix_elements::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, 0, 0);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
    ASSERT_EQ(testMpiTaskSequential.run(), true);
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);

    ASSERT_EQ(reference_sum[0], 0);
  }
}

TEST(TestMPITaskSequential, test_sum_with_single_element_matrix) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
     int n = 1;                
     int m = 1;               
    std::vector<int> global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);
    std::vector<int32_t> reference_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, n, m);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
    ASSERT_EQ(testMpiTaskSequential.run(), true);
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);

    int sum = chistov_a_sum_of_matrix_elements::classic_way(global_matrix, n, m);
    ASSERT_EQ(reference_sum[0], sum); 
  }
}

TEST(TestMPITaskSequential, throws_when_small_n_or_m) {
  boost::mpi::communicator world;
  int n, m; 
  if (world.rank() == 0) {
      n = 0;  
     m = 4;

    EXPECT_THROW({ chistov_a_sum_of_matrix_elements::getRandomMatrix( n, m); }, std::invalid_argument);

    m = 0;  
    EXPECT_THROW({ chistov_a_sum_of_matrix_elements::getRandomMatrix(1, m); }, std::invalid_argument);
  }
}

TEST(TestMPITaskSequential, test_wrong_validation) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int n, m; 
  if (world.rank() == 0) {
    n = 3;
     m = 4;
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataSeq->outputs_count.emplace_back(global_sum.size());
  }
  chistov_a_sum_of_matrix_elements::TestMPITaskSequential testMpiTaskSequential(taskDataSeq,n,m);
  ASSERT_EQ(testMpiTaskSequential.validation(), false);
}

TEST(TestMPITaskParallel, test_wrong_validation) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(2, 0); 

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int n, m;
  if (world.rank() == 0) {
    n = 3;
    m = 4;
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  
    chistov_a_sum_of_matrix_elements::TestMPITaskParallel TestMPITaskParallel(taskDataPar, n, m);

    ASSERT_EQ(TestMPITaskParallel.validation(), false);
  
  }
}

TEST(TestMPITaskParallel, test_sum1) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int n = 3; 
  int m = 4;

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel testMPITaskParallel(taskDataPar, n, m);
  ASSERT_TRUE(testMPITaskParallel.validation());
  ASSERT_TRUE(testMPITaskParallel.pre_processing());
  ASSERT_TRUE(testMPITaskParallel.run());
  ASSERT_TRUE(testMPITaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, n, m);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());


    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(TestMPITaskParallel, test_with_empty_matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel TestMPITaskParallel(taskDataPar, 0, 0);
  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);
  }
}

TEST(TestMPITaskParallel, test_with_large_matrix) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

   int n = 1000;
   int m = 1000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel TestMPITaskParallel(taskDataPar, n, m);
  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);
    for (int val : global_matrix) {
      reference_sum[0] += val;
    }
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}


TEST(TestMPITaskParallel, short_and_thick_test) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

  const int n = 100000;  
  const int m = 1; 

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel TestMPITaskParallel(taskDataPar, n, m);
  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);
    for (int val : global_matrix) {
      reference_sum[0] += val;
    }
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(TestMPITaskParallel, long_and_thin_test) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

  const int n = 1;
  const int m = 100000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel TestMPITaskParallel(taskDataPar, n, m);
  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);
    for (int val : global_matrix) {
      reference_sum[0] += val;
    }
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}


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
