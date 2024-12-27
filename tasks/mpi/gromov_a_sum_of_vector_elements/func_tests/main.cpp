#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/gromov_a_sum_of_vector_elements/include/ops_mpi.hpp"

namespace gromov_a_sum_of_vector_elements_mpi {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100, 100);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
}  // namespace gromov_a_sum_of_vector_elements_mpi

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Min1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {-10, -20, 0, 15, -30};
  std::vector<int32_t> global_min(1, std::numeric_limits<int32_t>::max());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "min");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, std::numeric_limits<int32_t>::max());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "min");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Min2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {-10, -17, 1, 19, 28};
  std::vector<int32_t> global_min(1, std::numeric_limits<int32_t>::max());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "min");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, std::numeric_limits<int32_t>::max());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "min");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Min3) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {-10, -20, 0, -30, 15};
  std::vector<int32_t> global_min(1, std::numeric_limits<int32_t>::max());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "min");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, std::numeric_limits<int32_t>::max());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "min");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Max1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, std::numeric_limits<int32_t>::min());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 250;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "max");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "max");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Max2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, std::numeric_limits<int32_t>::min());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 200;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "max");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "max");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Addition1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_add(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 200;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_add.data()));
    taskDataPar->outputs_count.emplace_back(global_add.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "add");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_add(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_add.data()));
    taskDataSeq->outputs_count.emplace_back(reference_add.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "add");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_add[0], global_add[0]);
  }
}

TEST(gromov_a_sum_of_vector_elements_mpi, Test_Addition2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_add(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 500;
    global_vec = gromov_a_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_add.data()));
    taskDataPar->outputs_count.emplace_back(global_add.size());
  }

  gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel MPISumOfVectorParallel(taskDataPar, "add");
  ASSERT_EQ(MPISumOfVectorParallel.validation(), true);
  MPISumOfVectorParallel.pre_processing();
  MPISumOfVectorParallel.run();
  MPISumOfVectorParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_add(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_add.data()));
    taskDataSeq->outputs_count.emplace_back(reference_add.size());

    // Create Task
    gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential MPISumOfVectorSequential(taskDataSeq, "add");
    ASSERT_EQ(MPISumOfVectorSequential.validation(), true);
    MPISumOfVectorSequential.pre_processing();
    MPISumOfVectorSequential.run();
    MPISumOfVectorSequential.post_processing();

    ASSERT_EQ(reference_add[0], global_add[0]);
  }
}
