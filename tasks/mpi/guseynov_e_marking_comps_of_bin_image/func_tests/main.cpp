#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/guseynov_e_marking_comps_of_bin_image/include/ops_mpi.hpp"

std::vector<int> getRandomBinImage(int r, int c) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(r * c);
  for (int i = 0; i < r * c; i++) {
    vec[i] = gen() % 2;
  }
  return vec;
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, fixed_validation_1) {
  boost::mpi::communicator world;
  const int rows = 100;
  const int cols = 100;
  std::vector<int> image(rows * cols, 0);
  image[50] = 10;
  std::vector<int> global_labeled_image(rows * cols);
  std::vector<int> expected_label(rows * cols, 2);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskParallel.validation());
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, fixed_validation_2) {
  boost::mpi::communicator world;
  const int rows = 100;
  const int cols = 0;
  std::vector<int> image(rows * cols, 0);
  std::vector<int> global_labeled_image(rows * cols);
  std::vector<int> expected_label(rows * cols, 2);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskParallel.validation());
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, fixed_test_1) {
  boost::mpi::communicator world;
  const int rows = 100;
  const int cols = 100;
  std::vector<int> image(rows * cols, 0);
  std::vector<int> global_labeled_image(rows * cols);
  std::vector<int> expected_label(rows * cols, 2);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_labeled_image, expected_label);
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, fixed_test_2) {
  boost::mpi::communicator world;
  const int rows = 100;
  const int cols = 100;
  std::vector<int> image(rows * cols, 1);
  std::vector<int> global_labeled_image(rows * cols);
  std::vector<int> expected_label(rows * cols, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_labeled_image, expected_label);
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, random_tes_25x25) {
  boost::mpi::communicator world;
  const int rows = 25;
  const int cols = 25;
  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = getRandomBinImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, random_tes_50x25) {
  boost::mpi::communicator world;
  const int rows = 50;
  const int cols = 25;
  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = getRandomBinImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, random_tes_50x50) {
  boost::mpi::communicator world;
  const int rows = 50;
  const int cols = 50;
  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = getRandomBinImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, random_tes_75x75) {
  boost::mpi::communicator world;
  const int rows = 75;
  const int cols = 75;
  std::vector<int> image;
  std::vector<int> global_labeled_image(rows * cols);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = getRandomBinImage(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_labeled_image.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_labeled_image(rows * cols);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_labeled_image.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Create Task
    guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_labeled_image, global_labeled_image);
  }
}
