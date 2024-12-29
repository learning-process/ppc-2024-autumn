// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/ermilova_d_Shell_sort_simple_merge/include/ops_mpi.hpp"

static std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Can_create_vector) {
  const int size_test = 10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_NO_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Cant_create_incorrect_vector) {
  const int size_test = -10;
  const int upper_border_test = 100;
  const int lower_border_test = -100;
  EXPECT_ANY_THROW(getRandomVector(size_test, upper_border_test, lower_border_test));
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 1;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_10) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 10;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_100) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 100;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_1000) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 1000;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_5000) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 5000;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_8) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 8;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_32) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 32;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_128) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 128;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_512) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 512;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_9) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 9;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_27) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 27;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_81) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 81;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_243) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 243;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_729) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 729;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_317) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 317;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_457) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 457;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_vec_809) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 809;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_reverse_sort_vec_100) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 100;
  bool is_resersed = true;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}

TEST(ermilova_d_Shell_sort_simple_merge_mpi, Test_reverse_sort_vec_347) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 347;
  bool is_resersed = true;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> sort_ref(input.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sort_ref.data()));
    taskDataSeq->outputs_count.emplace_back(sort_ref.size());

    // Create Task
    ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(sort_ref, output);
  }
}
