#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter/include/ops_mpi.hpp"

static std::vector<int> getRandomVector_(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_pipeline) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);
  ASSERT_EQ(testMpiTaskParallel.run(), true);
  ASSERT_EQ(testMpiTaskParallel.post_processing(), true);
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Validation_Wrong_Operation) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "min");

  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);
  ASSERT_EQ(testMpiTaskParallel.run(), true);
  ASSERT_EQ(testMpiTaskParallel.post_processing(), true);
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Operation_Plus) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::string operation = "+";

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel TestMPITaskParallel(taskDataPar, operation);

  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);
  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Operation_Minus) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::string operation = "-";

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel TestMPITaskParallel(taskDataPar, operation);

  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);
  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Operation_Max) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::string operation = "max";

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel TestMPITaskParallel(taskDataPar, operation);

  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  ASSERT_EQ(TestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(TestMPITaskParallel.run(), true);
  ASSERT_EQ(TestMPITaskParallel.post_processing(), true);
  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Uniform_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_(count_size_vector, 555, 555);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel TestMPITaskParallel(taskDataPar, "+");
  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  TestMPITaskParallel.pre_processing();
  TestMPITaskParallel.run();
  TestMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Generator) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;  // working vector
  int a = -100;
  int b = 100;
  int count_size_vector = 500;
  global_vec = getRandomVector_(count_size_vector, a, b);
  int max = *std::max_element(global_vec.begin(), global_vec.end());
  int min = *std::min_element(global_vec.begin(), global_vec.end());

  ASSERT_LE(max, b);
  ASSERT_GE(min, a);
  ASSERT_EQ(count_size_vector, (int)global_vec.size());
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Empty_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Single_Element_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 1;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel TestMPITaskParallel(taskDataPar, "+");
  ASSERT_EQ(TestMPITaskParallel.validation(), true);
  TestMPITaskParallel.pre_processing();
  TestMPITaskParallel.run();
  TestMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Sum_Manual_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {1, -12, 45, 33, 133, -100, 75, 221};  // working vector
  std::vector<int32_t> global_res(1, 0);                               // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 130;
    global_vec = getRandomVector_(count_size_vector, 0, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Diff_Manual_vec) {
  for (int i = 0; i < 10; i++) {
    boost::mpi::communicator world;
    std::vector<int> global_vec = {1, -12, 45, 33, 133, -100, 75, 221};  // working vector
    std::vector<int32_t> global_res(1, 0);                               // result vector

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      const int count_size_vector = 130;
      global_vec = getRandomVector_(count_size_vector, 0, 0);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
      taskDataPar->inputs_count.emplace_back(global_vec.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
      taskDataPar->outputs_count.emplace_back(global_res.size());
    }

    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
      taskDataSeq->inputs_count.emplace_back(global_vec.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
      taskDataSeq->outputs_count.emplace_back(check_vec.size());

      varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();

      ASSERT_EQ(check_vec[0], global_res[0]);
    }
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Max_Manual_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {1, -12, 45, 33, 133, -100, 75, 221};  // working vector
  std::vector<int32_t> global_res(1, 0);                               // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Sum_Zero_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 130;
    global_vec = getRandomVector_(count_size_vector, 0, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Diff_Zero_vec) {
  for (int i = 0; i < 10; i++) {
    boost::mpi::communicator world;
    std::vector<int> global_vec;            // working vector
    std::vector<int32_t> global_res(1, 0);  // result vector

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      const int count_size_vector = 130;
      global_vec = getRandomVector_(count_size_vector, 0, 0);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
      taskDataPar->inputs_count.emplace_back(global_vec.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
      taskDataPar->outputs_count.emplace_back(global_res.size());
    }

    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
      taskDataSeq->inputs_count.emplace_back(global_vec.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
      taskDataSeq->outputs_count.emplace_back(check_vec.size());

      varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();

      ASSERT_EQ(check_vec[0], global_res[0]);
    }
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Max_Zero_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 130;
    global_vec = getRandomVector_(count_size_vector, 0, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Sum_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, -200, -1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Sum_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, 1, 200);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Sum_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "+");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Diff_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, -200, -1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Diff_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, 1, 200);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Diff_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "-");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "-");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Max_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 130;
    global_vec = getRandomVector_(count_size_vector, -200, -1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Max_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, 1, 200);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Max_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 111;
    global_vec = getRandomVector_(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, "max");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, "max");
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(check_vec[0], global_res[0]);
  }
}
