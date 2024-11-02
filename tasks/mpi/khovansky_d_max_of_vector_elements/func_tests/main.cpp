// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/khovansky_d_max_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> khovansky_d_max_of_vector_elements_mpi::GetRandomVector(int sz, int left, int right) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(sz);
  for (int i = 0; i < sz; i++) {
    v[i] = gen() % (1 + right - left) + left;
  }
  return v;
}

TEST(khovansky_d_max_of_vector_elements, Test_Max) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    const int left = 0;
    const int right = 100;
    global_vec = khovansky_d_max_of_vector_elements_mpi::GetRandomVector(count_size_vector, left, right);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel maxOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(maxOfVectorMPIParalle.validation(), true);
  maxOfVectorMPIParalle.pre_processing();
  maxOfVectorMPIParalle.run();
  maxOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential maxOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(maxOfVectorMPISequential.validation(), true);
    maxOfVectorMPISequential.pre_processing();
    maxOfVectorMPISequential.run();
    maxOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(khovansky_d_max_of_vector_elements, Test_Max_LargeVector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10000;
    const int left = 0;
    const int right = 100;
    global_vec = khovansky_d_max_of_vector_elements_mpi::GetRandomVector(count_size_vector, left, right);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel maxOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(maxOfVectorMPIParalle.validation(), true);
  maxOfVectorMPIParalle.pre_processing();
  maxOfVectorMPIParalle.run();
  maxOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential maxOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(maxOfVectorMPISequential.validation(), true);
    maxOfVectorMPISequential.pre_processing();
    maxOfVectorMPISequential.run();
    maxOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(khovansky_d_max_of_vector_elements, Test_Max_Negative_Values) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    const int left = -100;
    const int right = -1;
    global_vec = khovansky_d_max_of_vector_elements_mpi::GetRandomVector(count_size_vector, left, right);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel maxOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(maxOfVectorMPIParalle.validation(), true);
  maxOfVectorMPIParalle.pre_processing();
  maxOfVectorMPIParalle.run();
  maxOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential maxOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(maxOfVectorMPISequential.validation(), true);
    maxOfVectorMPISequential.pre_processing();
    maxOfVectorMPISequential.run();
    maxOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(khovansky_d_max_of_vector_elements, Test_Max_RepeatingValues) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    global_vec = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    global_vec.resize(count_size_vector, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel maxOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(maxOfVectorMPIParalle.validation(), true);
  maxOfVectorMPIParalle.pre_processing();
  maxOfVectorMPIParalle.run();
  maxOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential maxOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(maxOfVectorMPISequential.validation(), true);
    maxOfVectorMPISequential.pre_processing();
    maxOfVectorMPISequential.run();
    maxOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(khovansky_d_max_of_vector_elements, Test_Max_Empty_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {};
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel maxOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(maxOfVectorMPIParalle.validation(), true);
  maxOfVectorMPIParalle.pre_processing();
  maxOfVectorMPIParalle.run();
  maxOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential maxOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(maxOfVectorMPISequential.validation(), true);
    maxOfVectorMPISequential.pre_processing();
    maxOfVectorMPISequential.run();
    maxOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}