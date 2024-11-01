#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/vladimirova_j_max_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> CreateVector(size_t size, size_t spread_of_val) {
  // Init value for input and output
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> v(size);
  for (size_t i = 0; i < size; i++) {
    v[i] = (random() % (2 * spread_of_val + 1)) - spread_of_val;
  }
  return v;
}

std::vector<std::vector<int>> CreateInputMatrix(size_t row_c, size_t col_c, size_t spread_of_val) {
  std::vector<std::vector<int>> m(row_c);
  for (size_t i = 0; i < row_c; i++) {
    m[i] = CreateVector(col_c, spread_of_val);
  }
  return m;
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_ValMatrix_0) {
  const size_t size = 0;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(size, size, spread);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_CanCreate_10) { EXPECT_NO_THROW(CreateInputMatrix(10, 10, 10)); }

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_SquareMatrix_1) {
  const size_t size = 1;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(size, size, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_Matrix_1_2) {
  const size_t row = 1;
  const size_t col = 2;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(row, col, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_Matrix_3_1) {
  const size_t row = 1;
  const size_t col = 2;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(row, col, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_SquareMatrix_10) {
  const size_t size = 10;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(size, size, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_SquareMatrix_50) {
  const size_t size = 50;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(size, size, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_SquareMatrix_100) {
  const size_t size = 200;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(size, size, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_Matrix_10_50) {
  const size_t row = 10;
  const size_t col = 50;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(row, col, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_Matrix_100_50) {
  const size_t row = 100;
  const size_t col = 50;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(row, col, spread);
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_SquareMatrix_50_WithSeveralMax) {
  const size_t size = 50;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(size, size, spread);
    global_matr[0][0] = spread;
    global_matr[5][25] = spread;
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(vladimirova_j_max_of_vector_elements_mpi, Test_Matrix_100_50_WithSeveralMax) {
  const size_t row = 100;
  const size_t col = 50;
  const size_t spread = 30;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(1, -((int)(spread + 10)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = CreateInputMatrix(row, col, spread);
    global_matr[25][10] = spread;
    global_matr[5][25] = spread;
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  vladimirova_j_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matr[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matr[i].data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    vladimirova_j_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}
