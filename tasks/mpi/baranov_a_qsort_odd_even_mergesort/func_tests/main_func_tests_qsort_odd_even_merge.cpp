#include <gtest/gtest.h>

#include "mpi/baranov_a_qsort_odd_even_mergesort/include/header_b_a_qsort_odd_even_merge.hpp"
template <typename tp, typename = std::enable_if_t<std::is_arithmetic_v<tp>>>
void get_rnd_vec(std::vector<tp> &vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  if constexpr (std::is_integral_v<tp>) {
    // Для целых чисел
    std::uniform_int_distribution<tp> dist(0, vec.size());
    std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
  } else if constexpr (std::is_floating_point_v<tp>) {
    // Для вещественных чисел
    std::uniform_real_distribution<tp> dist(0.0, vec.size());
    std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_8_int) {
  const int N = 8;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_100_int) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_10000_int) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_1000000_int) {
  const int N = 1000000;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_100_double) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_1000_double) {
  const int N = 1000;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_10000_double) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}

TEST(baranov_a_qsort_odd_even_merge, Test_sort_100_uint) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<uint32_t> arr(N);
  std::vector<uint32_t> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<uint32_t> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}
TEST(baranov_a_qsort_odd_even_merge, Test_sort_10000_uint) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<uint32_t> arr(N);
  std::vector<uint32_t> out_vec(N);

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<uint32_t> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) {
    std::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out_vec);
  }
}