#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter/include/ops_mpi.hpp"
const int repeat = 1;

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_pipeline) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -100, 100);
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

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Empty_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -100, 100);
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

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Sum_Zero_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 130;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, 0, 0);
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
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 130;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, 0, 0);
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

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Max_Zero_vec) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 130;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, 0, 0);
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
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -200, -1);
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
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, 1, 200);
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
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -100, 100);
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

///

TEST(varfolomeev_g_transfer_from_one_to_all_scatter_mpi, Test_Diff_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;            // working vector
  std::vector<int32_t> global_res(1, 0);  // result vector

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -200, -1);
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
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, 1, 200);
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
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -100, 100);
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
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -200, -1);
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
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, 1, 200);
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
    const int count_size_vector = 550;
    global_vec = varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(count_size_vector, -100, 100);
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