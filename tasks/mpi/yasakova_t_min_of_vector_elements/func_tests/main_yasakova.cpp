// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/yasakova_t_min_of_vector_elements/include/ops_mpi_yasakova.hpp"

TEST(yasakova_t_min_of_vector_elements_mpi, testFindMinimumIn1xNMatrix) {
  const int count_columns = 10; 
  const int gen_minimum = -500;
  const int gen_maximum = 500;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix; 
  std::vector<int32_t> global_minimum(1, INT_MAX); 
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix.resize(1,
                         yasakova_t_min_of_vector_elements_mpi::RandomVector(count_columns, gen_minimum, gen_maximum));
    for (unsigned int j = 0; j < global_matrix[0].size(); j++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&global_matrix[0][j]));
    taskDataPar->inputs_count.emplace_back(1); 
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minimum.data()));
    taskDataPar->outputs_count.emplace_back(global_minimum.size());
  }
  yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference_minimum(1, INT_MAX); 
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&global_matrix[0][0])); 
    taskDataSeq->inputs_count.emplace_back(1); 
    taskDataSeq->inputs_count.emplace_back(count_columns); 
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minimum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minimum.size());
    yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
        
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_minimum[0], global_minimum[0]);
  }
}

TEST(yasakova_t_min_of_vector_elements_mpi, testFindMinimumIn10x10Matrix) {
  const int count_rows = 10;
  const int count_columns = 10;
  const int gen_minimum = -500;
  const int gen_maximum = 500;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_minimum(1, INT_MAX);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        yasakova_t_min_of_vector_elements_mpi::RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minimum.data()));
    taskDataPar->outputs_count.emplace_back(global_minimum.size());
  }

  yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_minimum(1, INT_MAX);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minimum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minimum.size());

    yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minimum[0], global_minimum[0]);
  }
}

TEST(yasakova_t_min_of_vector_elements_mpi, testFindMinimumIn10x100Matrix) {
  const int count_rows = 10;
  const int count_columns = 100;
  const int gen_minimum = -500;
  const int gen_maximum = 500;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_minimum(1, INT_MAX);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        yasakova_t_min_of_vector_elements_mpi::RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minimum.data()));
    taskDataPar->outputs_count.emplace_back(global_minimum.size());
  }

  yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_minimum(1, INT_MAX);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minimum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minimum.size());

    yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minimum[0], global_minimum[0]);
  }
}

TEST(yasakova_t_min_of_vector_elements_mpi, testFindMinimumIn100x10Matrix) {
  const int count_rows = 100;
  const int count_columns = 10;
  const int gen_minimum = -500;
  const int gen_maximum = 500;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_minimum(1, INT_MAX);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        yasakova_t_min_of_vector_elements_mpi::RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minimum.data()));
    taskDataPar->outputs_count.emplace_back(global_minimum.size());
  }

  yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_minimum(1, INT_MAX);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minimum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minimum.size());

    yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minimum[0], global_minimum[0]);
  }
}

TEST(yasakova_t_minof_vector_elements_mpi, testFindMinimumIn100x100Matrix) {
  const int count_rows = 100;
  const int count_columns = 100;
  const int gen_minimum = -500;
  const int gen_maximum = 500;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_minimum(1, INT_MAX);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix =
        yasakova_t_min_of_vector_elements_mpi::RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minimum.data()));
    taskDataPar->outputs_count.emplace_back(global_minimum.size());
  }

  yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_minimum(1, INT_MAX);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_columns);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_minimum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_minimum.size());

    yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_minimum[0], global_minimum[0]);
  }
}