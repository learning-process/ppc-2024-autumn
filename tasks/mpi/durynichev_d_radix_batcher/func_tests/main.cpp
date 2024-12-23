#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/durynichev_d_radix_batcher/include/ops_mpi.hpp"

namespace durynichev_d_radix_batcher_mpi {

std::vector<double> generateRandomData(int array_size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-1000, 1000);

  std::vector<double> arr(array_size);
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] = dist(gen);
  }
  return arr;
}

void testTemplate(const std::vector<double>& input_array) {
  boost::mpi::communicator world;
  int array_size = input_array.size();
  std::vector<double> array;
  std::vector<double> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&array_size));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    array = input_array;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataPar->inputs_count.emplace_back(array.size());

    result.resize(array_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  auto taskParallel = std::make_shared<RadixBatcher>(taskDataPar);

  if (array_size < 1) {
    if (world.rank() == 0) {
      ASSERT_FALSE(taskParallel->validation());
    }
  } else {
    ASSERT_TRUE(taskParallel->validation());
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::sort(array.begin(), array.end());
      EXPECT_EQ(array, result);
    }
  }
}

}  // namespace durynichev_d_radix_batcher_mpi

TEST(durynichev_d_radix_batcher_mpi, sort_test_empty) {
  std::vector<double> empty_vector;

  EXPECT_NO_THROW(durynichev_d_radix_batcher_mpi::radixSortDouble(empty_vector.begin(), empty_vector.end()));
  EXPECT_TRUE(empty_vector.empty());
}

TEST(durynichev_d_radix_batcher_mpi, sort_test_right_work) {
  std::vector<double> data = {3.1, 2.4, 5.6, 1.2, 8.9, 7.3, 4.5, 6.7};
  std::vector<double> expected = {1.2, 2.4, 3.1, 4.5, 5.6, 6.7, 7.3, 8.9};

  durynichev_d_radix_batcher_mpi::radixSortDouble(data.begin(), data.end());
  EXPECT_EQ(data, expected);
}

TEST(durynichev_d_radix_batcher_mpi, positive_numbers_size_8) {
  durynichev_d_radix_batcher_mpi::testTemplate({2.0, 4.2, 1.3, 5.7, 0.1, 8.9, 7.4, 6.1});
}

TEST(durynichev_d_radix_batcher_mpi, negative_numbers_size_8) {
  durynichev_d_radix_batcher_mpi::testTemplate({-2.0, -4.2, -1.3, -5.7, -0.1, -8.9, -7.4, -6.1});
}

TEST(durynichev_d_radix_batcher_mpi, mixed_numbers_size_8) {
  durynichev_d_radix_batcher_mpi::testTemplate({-2.0, 4.2, -1.3, 5.7, 0.1, -8.9, 7.4, -6.1});
}

TEST(durynichev_d_radix_batcher_mpi, fractional_numbers_size_4) {
  durynichev_d_radix_batcher_mpi::testTemplate({0.2, 0.002, 0.0002, 0.02});
}

TEST(durynichev_d_radix_batcher_mpi, large_numbers_size_8) {
  durynichev_d_radix_batcher_mpi::testTemplate({2e11, 4e16, 1e13, 5e19, 2e10, 8e21, 7e12, 6e15});
}

TEST(durynichev_d_radix_batcher_mpi, small_numbers_size_8) {
  durynichev_d_radix_batcher_mpi::testTemplate({2e-11, 4e-16, 1e-13, 5e-19, 2e-10, 8e-21, 7e-12, 6e-15});
}

TEST(durynichev_d_radix_batcher_mpi, numbers_with_infinity_size_4) {
  durynichev_d_radix_batcher_mpi::testTemplate(
      {2.0, -2.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()});
}

TEST(durynichev_d_radix_batcher_mpi, identical_numbers_size_4) {
  durynichev_d_radix_batcher_mpi::testTemplate({3.3, 3.3, 3.3, 3.3});
}

TEST(durynichev_d_radix_batcher_mpi, empty_vector) { durynichev_d_radix_batcher_mpi::testTemplate({}); }

TEST(durynichev_d_radix_batcher_mpi, single_element) { durynichev_d_radix_batcher_mpi::testTemplate({24.0}); }

TEST(durynichev_d_radix_batcher_mpi, numbers_close_to_zero_size_4) {
  durynichev_d_radix_batcher_mpi::testTemplate({-0.000002, 0.0, 0.000002, 0.000003});
}

TEST(durynichev_d_radix_batcher_mpi, very_large_and_very_small_numbers_size_4) {
  durynichev_d_radix_batcher_mpi::testTemplate({2e19, 2e-19, -2e19, -2e-19});
}

TEST(durynichev_d_radix_batcher_mpi, random_vector_size_1) {
  auto vec = durynichev_d_radix_batcher_mpi::generateRandomData(1);
  durynichev_d_radix_batcher_mpi::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_mpi, random_vector_size_2) {
  auto vec = durynichev_d_radix_batcher_mpi::generateRandomData(2);
  durynichev_d_radix_batcher_mpi::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_mpi, random_vector_size_8) {
  auto vec = durynichev_d_radix_batcher_mpi::generateRandomData(8);
  durynichev_d_radix_batcher_mpi::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_mpi, random_vector_size_64) {
  auto vec = durynichev_d_radix_batcher_mpi::generateRandomData(64);
  durynichev_d_radix_batcher_mpi::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_mpi, random_vector_size_512) {
  auto vec = durynichev_d_radix_batcher_mpi::generateRandomData(512);
  durynichev_d_radix_batcher_mpi::testTemplate(vec);
}
