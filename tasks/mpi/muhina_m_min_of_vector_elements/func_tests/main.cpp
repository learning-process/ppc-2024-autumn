// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/muhina_m_min_of_vector_elements/include/ops_mpi.hpp"

TEST(muhina_m_min_of_vector_elements, Test_Min) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    const int min_val = 0;
    const int max_val = 100;
    global_vec = muhina_m_min_of_vector_elements_mpi::GetRandomVector(count_size_vector, min_val, max_val);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel minOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(minOfVectorMPIParalle.validation(), true);
  minOfVectorMPIParalle.pre_processing();
  minOfVectorMPIParalle.run();
  minOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential minOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(minOfVectorMPISequential.validation(), true);
    minOfVectorMPISequential.pre_processing();
    minOfVectorMPISequential.run();
    minOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(muhina_m_min_of_vector_elements, Test_Min_LargeVector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10000;
    const int min_val = 0;
    const int max_val = 100;
    global_vec = muhina_m_min_of_vector_elements_mpi::GetRandomVector(count_size_vector, min_val, max_val);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel minOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(minOfVectorMPIParalle.validation(), true);
  minOfVectorMPIParalle.pre_processing();
  minOfVectorMPIParalle.run();
  minOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential minOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(minOfVectorMPISequential.validation(), true);
    minOfVectorMPISequential.pre_processing();
    minOfVectorMPISequential.run();
    minOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(muhina_m_min_of_vector_elements, Test_Min_NegativeValues) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    const int min_val = -100;
    const int max_val = -10;
    global_vec = muhina_m_min_of_vector_elements_mpi::GetRandomVector(count_size_vector, min_val, max_val);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel minOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(minOfVectorMPIParalle.validation(), true);
  minOfVectorMPIParalle.pre_processing();
  minOfVectorMPIParalle.run();
  minOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential minOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(minOfVectorMPISequential.validation(), true);
    minOfVectorMPISequential.pre_processing();
    minOfVectorMPISequential.run();
    minOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(muhina_m_min_of_vector_elements, Test_Min_RepeatingValues) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    global_vec = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    global_vec.resize(count_size_vector, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  muhina_m_min_of_vector_elements_mpi::MinOfVectorMPIParallel minOfVectorMPIParalle(taskDataPar);
  ASSERT_EQ(minOfVectorMPIParalle.validation(), true);
  minOfVectorMPIParalle.pre_processing();
  minOfVectorMPIParalle.run();
  minOfVectorMPIParalle.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_min(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    // Create Task
    muhina_m_min_of_vector_elements_mpi::MinOfVectorMPISequential minOfVectorMPISequential(taskDataSeq);
    ASSERT_EQ(minOfVectorMPISequential.validation(), true);
    minOfVectorMPISequential.pre_processing();
    minOfVectorMPISequential.run();
    minOfVectorMPISequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}
