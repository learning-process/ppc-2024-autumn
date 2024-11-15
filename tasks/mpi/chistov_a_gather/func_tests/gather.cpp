#include "mpi/chistov_a_gather/include/gather.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

TEST(chistov_a_gather, returns_empty_vector_when_small_size_) {
  auto vector1 = chistov_a_gather::getRandomVector<int>(0);
  EXPECT_TRUE(vector1.empty());

  auto vector2 = chistov_a_gather::getRandomVector<int>(-1);
  EXPECT_TRUE(vector2.empty());
}

TEST(chistov_a_gather, test_gather_empty_vectors) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> local_vector;
    std::vector<int> gathered_data;
    local_vector = chistov_a_gather::getRandomVector<int>(0);

    ASSERT_FALSE(chistov_a_gather::gather<int>(world, local_vector, local_vector.size(), gathered_data, 0));
  }
}

TEST(chistov_a_gather, test_incorrect_root_process) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> local_vector;
    std::vector<int> gathered_data;
    local_vector = chistov_a_gather::getRandomVector<int>(1);

    ASSERT_FALSE(chistov_a_gather::gather<int>(world, local_vector, local_vector.size(), gathered_data, -1));
  }
}

TEST(chistov_a_gather, test_empty_task_data) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> local_vector;
    std::vector<int> gathered_data;

    ASSERT_FALSE(chistov_a_gather::gather<int>(world, local_vector, local_vector.size(), gathered_data, 0));
  }
}

TEST(chistov_a_gather, test_int_gather) {
  boost::mpi::communicator world;
  int count = 2;
  std::vector<int> local_vector = chistov_a_gather::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather, test_double_gather) {
  boost::mpi::communicator world;
  int count = 2;
  std::vector<double> local_vector = chistov_a_gather::getRandomVector<double>(count);
  std::vector<double> my_gathered_vector;
  std::vector<double> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather::gather<double>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather, test_float_gather) {
  boost::mpi::communicator world;
  int count = 2;
  std::vector<float> local_vector = chistov_a_gather::getRandomVector<float>(count);
  std::vector<float> my_gathered_vector;
  std::vector<float> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather::gather<float>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather, test_large_size) {
  boost::mpi::communicator world;
  int count = 100000;
  std::vector<int> local_vector = chistov_a_gather::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather, test_not_zero_root) {
  boost::mpi::communicator world;
  if (world.size() == 1) return;
  int count = 5;
  int root = 1;
  std::vector<int> local_vector = chistov_a_gather::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, root);
  chistov_a_gather::gather<int>(world, local_vector, count, my_gathered_vector, root);

  if (world.rank() == root) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}