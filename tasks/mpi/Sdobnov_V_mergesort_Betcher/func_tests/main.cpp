#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/Sdobnov_V_mergesort_Betcher/include/ops_mpi.hpp"

TEST(Sdobnov_V_mergesort_Betcher_par, InvalidInputCount) {
  boost::mpi::communicator world;

  int size = 10;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 9, 7, 2, 6, 4, 8};
  std::vector<int> expected_res = {0, 1, 2, 2, 3, 4, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, InvalidInput) {
  boost::mpi::communicator world;

  int size = 10;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 9, 7, 2, 6, 4, 8};
  std::vector<int> expected_res = {0, 1, 2, 2, 3, 4, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, InvalidOutputCount) {
  boost::mpi::communicator world;

  int size = 10;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 9, 7, 2, 6, 4, 8};
  std::vector<int> expected_res = {0, 1, 2, 2, 3, 4, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, InvalidOutput) {
  boost::mpi::communicator world;

  int size = 10;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 1, 0, 3, 9, 7, 2, 6, 4, 8};
  std::vector<int> expected_res = {0, 1, 2, 2, 3, 4, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, SortTest8) {
  boost::mpi::communicator world;

  int size = 8;
  std::vector<int> res(size, 0);
  std::vector<int> input = {2, 8, 3, 9, 5, 6, 4, 0};
  std::vector<int> expected_res = {0, 2, 3, 4, 5, 6, 8, 9};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(res[i], expected_res[i]);
    }
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, SortTestRand16) {
  boost::mpi::communicator world;

  int size = 16;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_par::generate_random_vector(size, 0, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_res = input;
    std::sort(expected_res.begin(), expected_res.end());
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(res[i], expected_res[i]);
    }
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, SortTestRand32) {
  boost::mpi::communicator world;

  int size = 32;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_par::generate_random_vector(size, 0, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_res = input;
    std::sort(expected_res.begin(), expected_res.end());
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(res[i], expected_res[i]);
    }
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, SortTestRand64) {
  boost::mpi::communicator world;

  int size = 64;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_par::generate_random_vector(size, 0, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_res = input;
    std::sort(expected_res.begin(), expected_res.end());
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(res[i], expected_res[i]);
    }
  }
}
