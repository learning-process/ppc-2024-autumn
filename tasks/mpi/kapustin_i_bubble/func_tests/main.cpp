#include <gtest/gtest.h>

#include "mpi/kapustin_i_bubble/include/avg_mpi.hpp"

TEST(kapustin_i_bubble_sort, simple_test) {
  boost::mpi::communicator world;

  std::vector<int> in = {5,  3,   8,  1,    9,  2,  4,  12,  14, 25, 345, 1986, 6666, 125, 306, 17,  6,   29, 57,
                         24, 500, 45, 1024, 70, 11, 77, 150, 39, 19, 999, 88,   213,  344, 278, 420, 555, 888};
  std::vector<int> out(in.size());

  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }

  kapustin_i_bubble_sort_mpi::BubbleSortMPI tmpTaskPar(tmpPar);

  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected = {1,   2,   3,   4,   5,   6,   8,   9,   11,   12,   14,  17,  19,
                                 24,  25,  29,  39,  45,  57,  70,  77,  88,   125,  150, 213, 278,
                                 306, 344, 345, 420, 500, 555, 888, 999, 1024, 1986, 6666};

    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_EQ(out[i], expected[i]);
    }
  }
}

TEST(kapustin_i_bubble_sort, pref_sort) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<int> out(in.size());

  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }

  kapustin_i_bubble_sort_mpi::BubbleSortMPI tmpTaskPar(tmpPar);

  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_EQ(out[i], expected[i]);
    }
  }
}

TEST(kapustin_i_bubble_sort, some_eq_val) {
  boost::mpi::communicator world;

  std::vector<int> in = {5, 1, 3, 5, 2, 3, 5};
  std::vector<int> out(in.size());

  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }

  kapustin_i_bubble_sort_mpi::BubbleSortMPI tmpTaskPar(tmpPar);

  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected = {1, 2, 3, 3, 5, 5, 5};

    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_EQ(out[i], expected[i]);
    }
  }
}

TEST(kapustin_i_bubble_sort, empty_input) {
  boost::mpi::communicator world;

  std::vector<int> in = {};
  std::vector<int> out(in.size());

  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }

  kapustin_i_bubble_sort_mpi::BubbleSortMPI tmpTaskPar(tmpPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(tmpTaskPar.validation());
  }
}

TEST(kapustin_i_bubble_sort, random_input) {
  boost::mpi::communicator world;

  const int data_size = 100;
  std::vector<int> in(data_size);
  std::generate(in.begin(), in.end(), []() { return rand() % 2000 - 1000; });
  std::vector<int> out(in.size());

  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }

  kapustin_i_bubble_sort_mpi::BubbleSortMPI tmpTaskPar(tmpPar);

  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(out, expected);
  }
}
