#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter_custom/include/ops_mpi.hpp"

static std::vector<int> getRandomVector_Custom(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_pipeline) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.run(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.post_processing(), true);
}

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Validation_Wrong_Operation) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "min");

  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), false);
  ASSERT_EQ(MyScatterTestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.run(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.post_processing(), true);
}

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Operation_Plus) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::string operation = "+";

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, operation);

  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.run(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.post_processing(), true);
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Operation_Minus) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::string operation = "-";

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, operation);

  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.run(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.post_processing(), true);
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Operation_Max) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::string operation = "max";

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, operation);

  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.pre_processing(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.run(), true);
  ASSERT_EQ(MyScatterTestMPITaskParallel.post_processing(), true);
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Uniform_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, 555, 555);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Empty_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 0;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Single_Element_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 1;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Sum_Zero_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, 0, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Diff_Zero_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, 0, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "-");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Max_Zero_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, 0, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "max");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();
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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Sum_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -200, -1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq, "+");
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    EXPECT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Sum_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, 1, 200);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> check_vec(1, 0);  // vector to check algorithm by seq. realisation

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());
    varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential TestMPITaskSequential(taskDataSeq, "+");
    ASSERT_EQ(TestMPITaskSequential.validation(), true);
    TestMPITaskSequential.pre_processing();
    TestMPITaskSequential.run();
    TestMPITaskSequential.post_processing();

    EXPECT_EQ(check_vec[0], global_res[0]);
  }
}

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Sum_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "+");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Diff_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -200, -1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "-");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Diff_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, 1, 200);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "-");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Diff_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "-");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Max_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -200, -1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "max");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Max_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, 1, 200);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "max");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

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

TEST(varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi, Test_CustomScatter_Max_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    global_vec = getRandomVector_Custom(count_size_vector, -100, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel MyScatterTestMPITaskParallel(
      taskDataPar, "max");
  ASSERT_EQ(MyScatterTestMPITaskParallel.validation(), true);
  MyScatterTestMPITaskParallel.pre_processing();
  MyScatterTestMPITaskParallel.run();
  MyScatterTestMPITaskParallel.post_processing();

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
