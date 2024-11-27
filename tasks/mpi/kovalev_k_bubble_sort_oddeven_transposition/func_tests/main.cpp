#include <gtest/gtest.h>

#include "mpi/kovalev_k_bubble_sort_oddeven_transposition/include/header.hpp"

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, zero_length) {
  std::vector<int> in;
  std::vector<int> out;
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<int> tmpTaskPar(tmpPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(tmpTaskPar.validation());
  }
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, Test_NoViol_300_int) {
  const size_t length = 300;
  std::srand(std::time(nullptr));
  const int alpha = rand();
  std::vector<int> in(length, alpha);
  std::vector<int> out(length);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<int> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    int count_viol = 0;
    for (size_t i = 0; i < length; i++) {
      if (out[i] != in[i]) count_viol++;
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, Test_300_int) {
  const size_t length = 300;
  std::srand(std::time(nullptr));
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) in[i] = rand() * pow(-1, rand());
  std::vector<int> out(length);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<int> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    std::sort(in.begin(), in.end(), [](int a, int b) { return a < b; });
    int count_viol = 0;
    for (size_t i = 0; i < length; i++) {
      if (out[i] != in[i]) count_viol++;
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, Test_30000_int) {
  const size_t length = 30000;
  std::srand(std::time(nullptr));
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) in[i] = rand() * pow(-1, rand());
  std::vector<int> out(length);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<int> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    std::sort(in.begin(), in.end(), [](int a, int b) { return a < b; });
    int count_viol = 0;
    for (size_t i = 0; i < length; i++) {
      if (out[i] != in[i]) count_viol++;
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, Test_300_double) {
  const size_t length = 300;
  std::srand(std::time(nullptr));
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  for (int i = 0; i <= length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<double> out(length);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<double> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    std::sort(in.begin(), in.end(), [](double a, double b) { return a < b; });
    int count_viol = 0;
    for (size_t i = 0; i < length; i++) {
      if (out[i] != in[i]) count_viol++;
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, Test_30000_double) {
  const size_t length = 30000;
  std::srand(std::time(nullptr));
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  for (int i = 0; i <= length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<double> out(length);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<double> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    std::sort(in.begin(), in.end(), [](double a, double b) { return a < b; });
    int count_viol = 0;
    for (size_t i = 0; i < length; i++) {
      if (out[i] != in[i]) count_viol++;
    }
    ASSERT_EQ(count_viol, 0);
  }
}