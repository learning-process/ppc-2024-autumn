// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/poroshin_v_cons_conv_hull_for_bin_image_comp/include/ops_mpi.hpp"

std::vector<int> poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(int m, int n) {
  std::vector<int> tmp(m * n);
  int n1 = std::max(n, m);
  int m1 = std::min(n, m);

  for (int &t : tmp) {
    t = (n1 + (std::rand() % (m1 - n1 + 7))) % 2;  // Bin image
  }

  return tmp;
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_empty_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 100;
  const int m = 100;
  std::vector<std::pair<int, int>> result;
  std::vector<int> tmp;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_0x100_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 100;
  const int m = 0;
  std::vector<std::pair<int, int>> result;
  std::vector<int> tmp;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_random_1x1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 1;
  const int m = 1;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    if (tmp[0] == 0) {
      std::vector<std::pair<int, int>> ans(m * n + 2);
      ASSERT_EQ(ans, result);
    } else {
      std::vector<std::pair<int, int>> ans = {{0, 0}, {0, 0}, {-1, -1}};
      std::vector<std::pair<int, int>> res(ans.size());
      std::copy(result.begin(), result.begin() + ans.size(), res.begin());
      ASSERT_EQ(ans, res);
    }
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_4x3_full_1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 3;
  const int m = 4;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp(m * n, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 0}, {3, 0}, {3, 2}, {0, 2}, {0, 0}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_2x5_full_1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 5;
  const int m = 2;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp(m * n, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 0}, {1, 0}, {1, 4}, {0, 4}, {0, 0}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_10x10_full_1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 10;
  const int m = 10;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp(m * n, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 0}, {9, 0}, {9, 9}, {0, 9}, {0, 0}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_11x11_test_1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 11;
  const int m = 11;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp = {0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
                          1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
                          0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
                          1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 2}, {4, 0},   {10, 0}, {10, 10}, {1, 10}, {0, 8},
                                            {0, 2}, {-1, -1}, {1, 0},  {2, 0},   {1, 0},  {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_11x11_test_2_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 11;
  const int m = 11;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp = {1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0,
                          0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
                          0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                          0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 0},  {0, 0},   {-1, -1}, {0, 2},  {2, 0}, {6, 0},   {9, 1},
                                            {10, 4}, {10, 5},  {1, 9},   {0, 7},  {0, 2}, {-1, -1}, {3, 8},
                                            {6, 7},  {9, 7},   {6, 10},  {3, 10}, {3, 8}, {-1, -1}, {8, 10},
                                            {9, 9},  {10, 10}, {8, 10},  {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_25x25_test_1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 25;
  const int m = 25;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp = {
      1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
      0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1,
      1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
      1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
      1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1,
      1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1,
      0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0,
      1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
      0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0,
      0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
      1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
      0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
      0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {
        {0, 0},   {21, 0},  {24, 14}, {24, 24}, {6, 24},  {4, 23},  {0, 13},  {0, 0},   {-1, -1}, {0, 15},  {2, 14},
        {5, 16},  {6, 21},  {1, 24},  {0, 24},  {0, 15},  {-1, -1}, {9, 0},   {9, 0},   {-1, -1}, {12, 23}, {12, 24},
        {12, 23}, {-1, -1}, {14, 22}, {14, 24}, {14, 22}, {-1, -1}, {16, 24}, {16, 24}, {-1, -1}, {22, 2},  {23, 0},
        {24, 0},  {24, 3},  {22, 2},  {-1, -1}, {23, 9},  {24, 9},  {23, 9},  {-1, -1}, {24, 6},  {24, 7},  {24, 6},
        {-1, -1}, {24, 11}, {24, 11}, {-1, -1}, {24, 18}, {24, 18}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_13x13_prime_test_1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 13;
  const int m = 13;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp = {1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0,
                          1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
                          0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                          1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                          1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0,
                          1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {
        {0, 0},  {2, 0},  {5, 1},   {6, 4},  {5, 5},   {3, 5},   {0, 2},   {0, 0},  {-1, -1}, {0, 5},
        {1, 5},  {1, 8},  {0, 8},   {0, 5},  {-1, -1}, {0, 11},  {9, 0},   {12, 0}, {12, 8},  {10, 12},
        {0, 12}, {0, 11}, {-1, -1}, {3, 7},  {3, 7},   {-1, -1}, {7, 0},   {7, 1},  {7, 0},   {-1, -1},
        {7, 11}, {8, 11}, {8, 12},  {7, 11}, {-1, -1}, {12, 12}, {12, 12}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_500x500_full_1_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 500;
  const int m = 500;
  std::vector<std::pair<int, int>> result(m * n + 2);
  std::vector<int> tmp(m * n, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
    test->inputs_count.emplace_back(m);
    test->inputs_count.emplace_back(n);
    test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> ans = {{0, 0}, {499, 0}, {499, 499}, {0, 499}, {0, 0}, {-1, -1}};
    std::vector<std::pair<int, int>> res(ans.size());
    std::copy(result.begin(), result.begin() + ans.size(), res.begin());
    ASSERT_EQ(ans, res);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_100x100_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 100;
  const int m = 100;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_250x250_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 250;
  const int m = 250;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_50x50_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 50;
  const int m = 50;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_300x17_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 17;
  const int m = 300;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_55x56_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 56;
  const int m = 55;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_23x343_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 343;
  const int m = 23;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_123x321_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 321;
  const int m = 123;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_101x102_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 102;
  const int m = 101;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_128x128_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 128;
  const int m = 128;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_19x91_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 91;
  const int m = 19;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_67x4_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 4;
  const int m = 67;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_143x22_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 22;
  const int m = 143;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_mpi, validation_and_check_222x99_random_image) {
  // Create data
  boost::mpi::communicator world;
  const int n = 99;
  const int m = 222;
  std::vector<std::pair<int, int>> result_1(m * n + 2);
  std::vector<int> tmp_1 = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::gen(m, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_1 = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    test_1->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_1.data()));
    test_1->inputs_count.emplace_back(m);
    test_1->inputs_count.emplace_back(n);
    test_1->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_1.data()));
  }
  poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel testMPITaskParallel(test_1);
  ASSERT_EQ(testMPITaskParallel.validation(), true);

  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::pair<int, int>> result_2(m * n + 2);
    std::vector<int> tmp_2 = tmp_1;
    std::shared_ptr<ppc::core::TaskData> test_2 = std::make_shared<ppc::core::TaskData>();
    test_2->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_2.data()));
    test_2->inputs_count.emplace_back(m);
    test_2->inputs_count.emplace_back(n);
    test_2->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_2.data()));
    poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential testMPITaskSequential(test_2);
    ASSERT_EQ(testMPITaskSequential.validation(), true);

    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
    ASSERT_EQ(result_1, result_2);
  }
}