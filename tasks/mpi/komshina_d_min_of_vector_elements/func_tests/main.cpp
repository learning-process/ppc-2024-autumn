#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/komshina_d_min_of_vector_elements/include/ops_mpi.hpp"

TEST(komshina_d_min_of_vector_elements_mpi, Test_Min_1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 500000;
    const int start_value = 1000000;
    const int decrement = 10;
    global_vec.resize(count);
    for (int i = 0; i < count; ++i) {
      global_vec[i] = start_value - i * decrement;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel minOfVectorElementTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    std::cout << "Running parallel task pre_processing..." << std::endl;
  }
  ASSERT_EQ(minOfVectorElementTaskParallel.validation(), true);
  minOfVectorElementTaskParallel.pre_processing();
  minOfVectorElementTaskParallel.run();
  minOfVectorElementTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(
        taskDataSeq);
    ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
    minOfVectorElementTaskSequential.pre_processing();
    minOfVectorElementTaskSequential.run();
    minOfVectorElementTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(komshina_d_min_of_vector_elements_mpi, Test_Min_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 5000000;
    const int start_value = -1;
    const int decrement = 100;
    global_vec.resize(count);
    for (int i = 0; i < count; ++i) {
      global_vec[i] = start_value - i * decrement;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel minOfVectorElementTaskParallel(taskDataPar);
  ASSERT_EQ(minOfVectorElementTaskParallel.validation(), true);
  minOfVectorElementTaskParallel.pre_processing();
  minOfVectorElementTaskParallel.run();
  minOfVectorElementTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(
        taskDataSeq);
    ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
    minOfVectorElementTaskSequential.pre_processing();
    minOfVectorElementTaskSequential.run();
    minOfVectorElementTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(komshina_d_min_of_vector_elements_mpi, Test_Min_3) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 10;
    global_vec.resize(count, 0);
    for (int i = 1; i < count; i += 1) {
      global_vec[i] = INT_MIN;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel minOfVectorElementTaskParallel(taskDataPar);
  ASSERT_EQ(minOfVectorElementTaskParallel.validation(), true);
  minOfVectorElementTaskParallel.pre_processing();
  minOfVectorElementTaskParallel.run();
  minOfVectorElementTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(
        taskDataSeq);
    ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
    minOfVectorElementTaskSequential.pre_processing();
    minOfVectorElementTaskSequential.run();
    minOfVectorElementTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(komshina_d_min_of_vector_elements_mpi, Test_Min_4) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 1000;
    const int value = 42;
    global_vec.resize(count, value);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel minOfVectorElementTaskParallel(taskDataPar);
  ASSERT_EQ(minOfVectorElementTaskParallel.validation(), true);
  minOfVectorElementTaskParallel.pre_processing();
  minOfVectorElementTaskParallel.run();
  minOfVectorElementTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(1, 42);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(
        taskDataSeq);
    ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
    minOfVectorElementTaskSequential.pre_processing();
    minOfVectorElementTaskSequential.run();
    minOfVectorElementTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}

TEST(komshina_d_min_of_vector_elements_mpi, Empty_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec = {};
  std::vector<int32_t> global_min(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel minOfVectorElementTaskParallel(taskDataPar);
  ASSERT_EQ(minOfVectorElementTaskParallel.validation(), true);
  minOfVectorElementTaskParallel.pre_processing();
  minOfVectorElementTaskParallel.run();
  minOfVectorElementTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_min(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(
        taskDataSeq);
    ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
    minOfVectorElementTaskSequential.pre_processing();
    minOfVectorElementTaskSequential.run();
    minOfVectorElementTaskSequential.post_processing();

    ASSERT_EQ(reference_min[0], global_min[0]);
  }
}