#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/task/include/task.hpp"
#include "mpi/lavrentyev_a_radix_sort_simple_merge/include/ops_mpi.hpp"

namespace lavrentyev_a_radix_sort_simple_merge_mpi {

std::vector<double> generate_random_vector(int sz, double min_val = -100.0, double max_val = 100.0) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  std::vector<double> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

void template_test(const std::vector<double>& input_data) {
  boost::mpi::communicator world;
  std::vector<double> data = input_data;
  std::vector<double> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(data.size());

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));

    result_data.resize(data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  }

  auto taskParallel = std::make_shared<RadixSimpleMerge>(taskDataPar);

  bool success = taskParallel->validation();
  boost::mpi::broadcast(world, success, 0);
  if (success) {
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::sort(data.begin(), data.end());
      EXPECT_EQ(data, result_data);
    }
  }
}

}  // namespace lavrentyev_a_radix_sort_simple_merge_mpi

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Simple_Positive_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({1.0, 3.5, 2.2, 4.8, 0.0});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Simple_Negative_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({-1.0, -3.5, -2.2, -4.8, 0.0});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Mixed_Positive_and_Negative_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({-1.0, 3.5, -2.2, 4.8, 0.0});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Fractional_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({0.1, 0.001, 0.0001, 1.0, 0.01});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Large_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({1e10, 3e15, 2e12, 4e18, 1e9});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Small_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({1e-10, 3e-15, 2e-12, 4e-18, 1e-9});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Numbers_with_Infinity) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(
      {1.0, -1.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Identical_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({5.5, 5.5, 5.5, 5.5});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Empty_Vector) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Single_Element) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({42.0});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Numbers_Close_to_Zero) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({-0.000001, 0.0, 0.000001});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Very_Large_and_Very_Small_Numbers) {
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test({1e18, 1e-18, -1e18, -1e-18});
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_1) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(2);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_2) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(4);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_3) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(8);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_4) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(16);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_5) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(32);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_6) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(64);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_7) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(128);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_8) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(256);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_9) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(512);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_2_pow_10) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(1024);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_3_pow_1) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(3);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_3_pow_2) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(9);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_3_pow_3) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(27);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_3_pow_4) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(81);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_3_pow_5) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(243);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_3_pow_6) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(729);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_Prime_7) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(7);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_Prime_13) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(13);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_Prime_17) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(17);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_Prime_23) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(23);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_Prime_47) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(47);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_Prime_59) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(59);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_10) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(10);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_100) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(100);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_1000) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(1000);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Negative_Values) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(50, -500.0, -1.0);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Positive_Values) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(50, 1.0, 500.0);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_With_Large_Range) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(100, -1e6, 1e6);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_With_Small_Range) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(100, -0.001, 0.001);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_With_Zeros) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(20, 0.0, 0.0);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_Size_1) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(1);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, Random_Vector_With_Duplicate_Values) {
  auto vec = lavrentyev_a_radix_sort_simple_merge_mpi::generate_random_vector(100, 10.0, 10.0);
  lavrentyev_a_radix_sort_simple_merge_mpi::template_test(vec);
}
