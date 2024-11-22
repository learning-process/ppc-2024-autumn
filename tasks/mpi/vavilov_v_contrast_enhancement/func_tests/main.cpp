#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "mpi/vavilov_v_contrast_enhancement/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

std::vector<uint8_t> generate_random_vector(size_t size, uint8_t min_value, uint8_t max_value) {
  std::vector<uint8_t> vec(size);
  for (auto &val : vec) {
    val = static_cast<uint8_t>(min_value + std::rand() % (max_value - min_value + 1));
  }
  return vec;
}

TEST(vavilov_v_contrast_enhancement_mpi, ValidInput) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();
    std::vector<uint8_t> input = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(vavilov_v_contrast_enhancement_mpi, CorrectOutputSize) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<uint8_t> input = {10, 20, 30, 40, 50};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());

    vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }
}

TEST(vavilov_v_contrast_enhancement_mpi, IncorrectOutputSize) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<uint8_t> input = {10, 20, 30, 40, 50};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());

    std::vector<uint8_t> output(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    taskDataPar->outputs_count.emplace_back(output.size() - 1);

    vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(vavilov_v_contrast_enhancement_mpi, NormalContrastEnhancement) {
  mpi::environment env;
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<uint8_t> input = {10, 20, 30, 40, 50};
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs_count.emplace_back(input.size());
  std::vector<uint8_t> output(input.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<uint8_t> expected_output = {0, 63, 127, 191, 255};
    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_contrast_enhancement_mpi, RandomContrastEnhancement) {
  mpi::environment env;
  mpi::communicator world;

  const size_t size = 1000;
  const uint8_t min_value = 0;
  const uint8_t max_value = 255;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = generate_random_vector(size, min_value, max_value);
  std::vector<uint8_t> output(input.size());
  std::vector<uint8_t> expected_output(input.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataPar->inputs_count.emplace_back(input.size());

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataPar->outputs_count.emplace_back(output.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_output.data()));
  taskDataSeq->outputs_count.emplace_back(expected_output.size());

  vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    vavilov_v_contrast_enhancement_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    EXPECT_EQ(output, expected_output);
  }
}

TEST(vavilov_v_contrast_enhancement_mpi, SingleValueInput) {
  mpi::environment env;
  mpi::communicator world;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<uint8_t> input = {128, 128, 128, 128, 128};
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs_count.emplace_back(input.size());
  std::vector<uint8_t> output(input.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  vavilov_v_contrast_enhancement_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<uint8_t> expected_output = {0, 0, 0, 0, 0};
    EXPECT_EQ(output, expected_output);
  }
}
