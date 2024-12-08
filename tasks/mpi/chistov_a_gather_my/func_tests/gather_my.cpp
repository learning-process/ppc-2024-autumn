#include "mpi/chistov_a_gather_my/include/gather_my.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

TEST(chistov_a_gather_my, test_empty_vector) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> local_vector;
    std::vector<int> gathered_data;
    local_vector = chistov_a_gather_my::getRandomVector<int>(0);

    ASSERT_FALSE(chistov_a_gather_my::gather<int>(world, local_vector, local_vector.size(), gathered_data, 0));
  }
}

TEST(chistov_a_gather_my, test_incorrect_root_process) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> local_vector;
    std::vector<int> gathered_data;
    local_vector = chistov_a_gather_my::getRandomVector<int>(1);

    ASSERT_FALSE(chistov_a_gather_my::gather<int>(world, local_vector, local_vector.size(), gathered_data, -1));
  }
}

TEST(chistov_a_gather_my, test_empty_task_data) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<int> local_vector;
    std::vector<int> gathered_data;

    ASSERT_FALSE(chistov_a_gather_my::gather<int>(world, local_vector, local_vector.size(), gathered_data, 0));
  }
}

TEST(chistov_a_gather_my, test_int_gather) {
  boost::mpi::communicator world;
  int count = 2;
  std::vector<int> local_vector = chistov_a_gather_my::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_double_gather) {
  boost::mpi::communicator world;
  int count = 2;
  std::vector<double> local_vector = chistov_a_gather_my::getRandomVector<double>(count);
  std::vector<double> my_gathered_vector;
  std::vector<double> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<double>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_float_gather) {
  boost::mpi::communicator world;
  int count = 2;
  std::vector<float> local_vector = chistov_a_gather_my::getRandomVector<float>(count);
  std::vector<float> my_gathered_vector;
  std::vector<float> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<float>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_char_gather) {
  boost::mpi::communicator world;
  int count = 2;
  std::vector<char> local_vector = chistov_a_gather_my::getRandomVector<char>(count);
  std::vector<char> my_gathered_vector;
  std::vector<char> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<char>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_large_size) {
  boost::mpi::communicator world;
  int count = 100000;
  std::vector<int> local_vector = chistov_a_gather_my::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_not_zero_root) {
  boost::mpi::communicator world;
  if (world.size() == 1) return;
  int count = 5;
  int root = 1;
  std::vector<int> local_vector = chistov_a_gather_my::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, root);
  chistov_a_gather_my::gather<int>(world, local_vector, count, my_gathered_vector, root);

  if (world.rank() == root) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_gather_fixed_values) {
  boost::mpi::communicator world;
  int count = 2;

  std::vector<int> local_vector;
  if (world.rank() == 0) {
    local_vector = {1, 2};
  } else {
    local_vector = {3, 4};
  }

  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;

  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_gather_different_values) {
  boost::mpi::communicator world;
  int count = 3;

  std::vector<int> local_vector = {world.rank(), world.rank() + 1, world.rank() + 2};
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;

  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_EQ(my_gathered_vector.size(), mpi_gathered_data.size());
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_count_is_a_powers_of_two) {
  boost::mpi::communicator world;
  int count = 1024;
  std::vector<int> local_vector = chistov_a_gather_my::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_count_is_a_prime_number_gather) {
  boost::mpi::communicator world;
  int count = 563;
  std::vector<int> local_vector = chistov_a_gather_my::getRandomVector<int>(count);
  std::vector<int> my_gathered_vector;
  std::vector<int> mpi_gathered_data;
  boost::mpi::gather(world, local_vector.data(), count, mpi_gathered_data, 0);
  chistov_a_gather_my::gather<int>(world, local_vector, count, my_gathered_vector, 0);

  if (world.rank() == 0) {
    ASSERT_TRUE(std::is_permutation(my_gathered_vector.begin(), my_gathered_vector.end(), mpi_gathered_data.begin()));
  }
}