// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kozlova_e_radix_batcher_sort/include/ops_mpi.hpp"

namespace kozlova_e_utility_functions {
std::vector<double> generate_random_double_vector(size_t size, double min_val, double max_val) {
  std::vector<double> result(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  for (auto& value : result) {
    value = dist(gen);
  }
  return result;
}
}  // namespace kozlova_e_utility_functions

TEST(kozlova_e_radix_batcher_sort_mpi, test_simple_mas) {
  boost::mpi::communicator world;
  std::vector<double> mas = {-12.34, 45.67, 0.0, -98.76, 32.1, -0.01, 23.45, -56.78, 9.99, 0.001};
  std::vector<double> resMPI(10, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ(10, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(mas.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential testMPITaskSequential(taskDataSeq);

    ASSERT_EQ(testMPITaskSequential.validation(), true);
    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();

    ASSERT_EQ(resSEQ, resMPI);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_sort_empty) {
  boost::mpi::communicator world;
  std::vector<double> mas = {};
  std::vector<double> resMPI;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(resMPI.empty());
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_single_element) {
  boost::mpi::communicator world;
  std::vector<double> mas = {42.0};
  std::vector<double> resMPI(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(mas, resMPI);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_incorrect_size) {
  boost::mpi::communicator world;
  std::vector<double> mas = {1.1, 2.2, 3.3};
  std::vector<double> resMPI(2);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_large_mas) {
  boost::mpi::communicator world;
  const int size = 10000;
  double min_val = -100.0;
  double max_val = 100.0;
  std::vector<double> mas(size);
  std::vector<double> resMPI(size);

  mas = kozlova_e_utility_functions::generate_random_double_vector(size, min_val, max_val);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ = mas;
    std::sort(resSEQ.begin(), resSEQ.end());

    ASSERT_EQ(resSEQ, resMPI);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_same_elements) {
  boost::mpi::communicator world;
  std::vector<double> mas = {0.000, 0.00, 0};
  std::vector<double> resMPI(3);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ(3, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(mas.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential testMPITaskSequential(taskDataSeq);

    ASSERT_EQ(testMPITaskSequential.validation(), true);
    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();

    ASSERT_EQ(resSEQ, resMPI);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_identical_elements_with_diff_signs) {
  boost::mpi::communicator world;
  std::vector<double> mas = {0.5, -0.5};
  std::vector<double> resMPI(2);
  std::vector<double> expected = {-0.5, 0.5};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ(2);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(mas.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential testMPITaskSequential(taskDataSeq);

    ASSERT_EQ(testMPITaskSequential.validation(), true);
    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();

    ASSERT_EQ(resSEQ, resMPI);
    ASSERT_EQ(resSEQ, expected);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_reverse_sort) {
  boost::mpi::communicator world;
  std::vector<double> mas = {5.1, 3.22, -3.34, -5.54};
  std::vector<double> resMPI(4);
  std::vector<double> expected = {-5.54, -3.34, 3.22, 5.1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ(4);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(mas.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resSEQ.size());

    kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential testMPITaskSequential(taskDataSeq);

    ASSERT_EQ(testMPITaskSequential.validation(), true);
    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();

    ASSERT_EQ(resSEQ, resMPI);
    ASSERT_EQ(resSEQ, expected);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_mas_2_n) {
  boost::mpi::communicator world;
  const int size = 1024;
  double min_val = -100.0;
  double max_val = 100.0;
  std::vector<double> mas(size);
  std::vector<double> resMPI(size);

  mas = kozlova_e_utility_functions::generate_random_double_vector(size, min_val, max_val);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ = mas;
    std::sort(resSEQ.begin(), resSEQ.end());

    ASSERT_EQ(resSEQ, resMPI);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_mas_3_n) {
  boost::mpi::communicator world;
  const int size = 2187;
  double min_val = -100.0;
  double max_val = 100.0;
  std::vector<double> mas(size);
  std::vector<double> resMPI(size);

  mas = kozlova_e_utility_functions::generate_random_double_vector(size, min_val, max_val);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ = mas;
    std::sort(resSEQ.begin(), resSEQ.end());

    ASSERT_EQ(resSEQ, resMPI);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_mas_prime_value_size) {
  boost::mpi::communicator world;
  const int size = 97;
  double min_val = -100.0;
  double max_val = 100.0;
  std::vector<double> mas(size);
  std::vector<double> resMPI(size);

  mas = kozlova_e_utility_functions::generate_random_double_vector(size, min_val, max_val);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mas.data()));
    taskDataPar->inputs_count.emplace_back(mas.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSEQ = mas;
    std::sort(resSEQ.begin(), resSEQ.end());

    ASSERT_EQ(resSEQ, resMPI);
  }
}