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

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq, n,m);
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

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());  
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(empty_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(empty_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
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

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
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
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  
    chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);

    ASSERT_EQ(TestMPITaskParallel.validation(), false);
  
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_int_sum_parallell) {
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int n = 3; 
  int m = 4;

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> testMPITaskParallel(taskDataPar, n, m);
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

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<int> testMpiTaskSequential(taskDataSeq, n, m);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());


    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_double_sum_parallell) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  std::vector<double> global_sum(1, 0.0);  

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int n = 3;
  int m = 4;

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<double>(n, m);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size()); 

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());  
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<double> testMPITaskParallel(taskDataPar, n, m);
  ASSERT_TRUE(testMPITaskParallel.validation());
  ASSERT_TRUE(testMPITaskParallel.pre_processing());
  ASSERT_TRUE(testMPITaskParallel.run());
  ASSERT_TRUE(testMPITaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<double> reference_sum(1, 0.0);  

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());  

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());  

    chistov_a_sum_of_matrix_elements::TestMPITaskSequential<double> testMpiTaskSequential(taskDataSeq, n, m);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_NEAR(reference_sum[0], global_sum[0], 1e-6);  
  }
}


TEST(chistov_a_sum_of_matrix_elements, test_with_empty_matrix_parallell) {
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

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, 0, 0);
  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_with_large_matrix_parallell) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

   int n = 1000;
   int m = 1000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);
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


TEST(chistov_a_sum_of_matrix_elements, short_and_thick_test_parallell) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

  const int n = 100000;  
  const int m = 1; 

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);
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

TEST(chistov_a_sum_of_matrix_elements, long_and_thin_test_parallell) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);

  const int n = 1;
  const int m = 100000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int> TestMPITaskParallel(taskDataPar, n, m);
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
