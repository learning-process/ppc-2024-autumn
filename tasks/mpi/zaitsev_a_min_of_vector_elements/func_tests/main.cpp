// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zaitsev_a_min_of_vector_elements/include/ops_mpi.hpp"

TEST(zaitsev_a_min_of_vector_elements_mpi, test_case_even_length_vector) {
  const int extrema = -1;
  const int minRangeValue = 0;
  const int maxRangeValue = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, maxRangeValue + 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10e2;
    global_vec = zaitsev_a_min_of_vector_elements_mpi::getRandomVector(count_size_vector, minRangeValue, maxRangeValue);
    global_vec[global_vec.size() / 2] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    // Create Task
    zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);

    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference[0], global_min[0]);
  }
}

TEST(zaitsev_a_min_of_vector_elements_mpi, test_case_odd_length_vector) {
  const int extrema = -1;
  const int minRangeValue = 0;
  const int maxRangeValue = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, maxRangeValue + 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10e2 + 1;
    global_vec = zaitsev_a_min_of_vector_elements_mpi::getRandomVector(count_size_vector, minRangeValue, maxRangeValue);
    global_vec[global_vec.size() / 2] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    // Create Task
    zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference[0], global_min[0]);
  }
}

TEST(zaitsev_a_min_of_vector_elements_mpi, test_case_singleton_vector) {
  const int extrema = -1;
  const int minRangeValue = 0;
  const int maxRangeValue = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, extrema);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1;
    global_vec = zaitsev_a_min_of_vector_elements_mpi::getRandomVector(count_size_vector, minRangeValue, maxRangeValue);
    global_vec[0] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(1, maxRangeValue + 1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    // Create Task
    zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference[0], global_min[0]);
  }
}

TEST(zaitsev_a_min_of_vector_elements_mpi, test_case_empty_vector) {
  const int minRangeValue = 0;
  const int maxRangeValue = 1000;

  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, maxRangeValue + 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    global_vec = zaitsev_a_min_of_vector_elements_mpi::getRandomVector(count_size_vector, minRangeValue, maxRangeValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    // Create Task
    zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}