#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/guseynov_e_my_scatter/include/ops_mpi.hpp"

std::vector<int> getRandomVectorMyScatter(int sz, int minValue, int maxValue) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(minValue, maxValue);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(guseynov_e_my_scatter_mpi, Test_seq_array_10) {
  const int count = 10;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  guseynov_e_my_scatter_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(guseynov_e_my_scatter_mpi, Test_seq_fixed_array_20) {
  const int count = 20;
  int sum = 0;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(1, 0);

  for (int i = 0; i < count; i++) {
    in[i] = -i;
    sum -= i;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  guseynov_e_my_scatter_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(sum, out[0]);
}

TEST(guseynov_e_my_scatter_mpi, Test_seq_fixed_array_40) {
  const int count = 40;
  int sum = 0;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(1, 0);

  for (int i = 0; i < count; i++) {
    in[i] = i;
    sum += i;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  guseynov_e_my_scatter_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(sum, out[0]);
}

TEST(guseynov_e_my_scatter_mpi, Test_seq_empty_array) {
  const int count = 0;

  // Create data
  std::vector<int> in;
  std::vector<int> out(1, -1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  guseynov_e_my_scatter_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(guseynov_e_my_scatter_mpi, Test_my_scatter_empty_array) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 0;
    const int minValue = -50;
    const int maxValue = 50;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_my_scatter_array_57) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 57;
    const int minValue = -50;
    const int maxValue = 50;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_boost_scatter_fixed_array_70) {
  boost::mpi::communicator world;
  const int vector_size = 70;
  std::vector<int> global_vec(vector_size);
  for (int i = 0; i < vector_size; i++) {
    global_vec[i] = i;
  }
  std::vector<int32_t> global_res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_my_scatter_array_79) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 79;
    const int minValue = -50;
    const int maxValue = 50;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_my_scatter_array_100) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 100;
    const int minValue = -50;
    const int maxValue = 50;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_my_scatter_array_153) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 153;
    const int minValue = -50;
    const int maxValue = 50;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_my_scatter_array_645) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 79;
    const int minValue = -50;
    const int maxValue = 50;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_boost_scatter_only_negatives_array_1000) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 1000;
    const int minValue = -100;
    const int maxValue = -1;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, 1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_my_scatter_mpi, Test_boost_scatter_only_positive_array_1000) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_res(1, -1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int vector_size = 1000;
    const int minValue = 1;
    const int maxValue = 100;
    global_vec = getRandomVectorMyScatter(vector_size, minValue, maxValue);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_my_scatter_mpi::MyScatterTestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_my_scatter_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}