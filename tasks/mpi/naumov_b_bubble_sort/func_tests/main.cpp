#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "mpi/naumov_b_bubble_sort/include/ops_mpi.hpp"

TEST(naumov_b_bubble_sort_mpi, Test_10_int) {
  const size_t length = 10;
  std::vector<int> in(length);
  std::vector<int> out(length);

  // Генерация случайных чисел
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(-1000, 1000);
  for (size_t i = 0; i < length; ++i) {
    in[i] = distribution(generator);
  }

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  if (in.empty() || out.empty()) {
    std::cerr << "Error: Input or output vectors are empty!" << std::endl;
  }

  if (world.rank() == 0) {
    try {
      tmpPar->inputs_count.emplace_back(in.size());
      tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
      tmpPar->outputs_count.emplace_back(out.size());
    } catch (const std::exception &e) {
      std::cerr << "Error while initializing tmpPar: " << e.what() << std::endl;
    }
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel tmpTaskPar(tmpPar);

  ASSERT_TRUE(tmpTaskPar.validation());

  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();

  std::vector<int> reference = in;
  std::sort(reference.begin(), reference.end());
  ASSERT_EQ(out, reference);
}

// TEST(naumov_b_bubble_sort_mpi, Test_100_int) {
//   const size_t length = 100;
//   std::vector<int> in(length);
//   std::vector<int> out(length);

//   std::mt19937 generator(std::random_device{}());
//   std::uniform_int_distribution<int> distribution(-1000, 1000);
//   for (size_t i = 0; i < length; ++i) {
//     in[i] = distribution(generator);
//   }

//   boost::mpi::communicator world;
//   std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

//   if (in.empty() || out.empty()) {
//     std::cerr << "Error: Input or output vectors are empty!" << std::endl;
//   }

//   if (world.rank() == 0) {
//     try {
//       tmpPar->inputs_count.emplace_back(in.size());
//       tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//       tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//       tmpPar->outputs_count.emplace_back(out.size());
//     } catch (const std::exception &e) {
//       std::cerr << "Error while initializing tmpPar: " << e.what() << std::endl;
//     }
//   }

//   naumov_b_bubble_sort_mpi::TestMPITaskParallel tmpTaskPar(tmpPar);

//   ASSERT_TRUE(tmpTaskPar.validation());

//   tmpTaskPar.pre_processing();
//   tmpTaskPar.run();
//   tmpTaskPar.post_processing();

//   std::vector<int> reference = in;
//   std::sort(reference.begin(), reference.end());
//   ASSERT_EQ(out, reference);
// }

TEST(naumov_b_bubble_sort_mpi, Test_empty_array) {
  const size_t length = 0;
  std::vector<int> in(length);
  std::vector<int> out(length);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  tmpPar->inputs_count.emplace_back(in.size());
  tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  tmpPar->outputs_count.emplace_back(out.size());

  naumov_b_bubble_sort_mpi::TestMPITaskParallel tmpTaskPar(tmpPar);

  ASSERT_FALSE(tmpTaskPar.validation());
}
