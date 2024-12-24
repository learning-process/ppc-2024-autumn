#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/gnitienko_k_contrast_enhancement/include/ops_mpi.hpp"

namespace gnitienko_k_functions {
std::vector<uint8_t> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<uint8_t> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 256;
  }
  return vec;
}
}  // namespace gnitienko_k_functions

TEST(gnitienko_k_contrast_enhancement_mpi, Test_grayscale_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> img = {12, 24, 85, 100};
  double contrast_factor = 1.5;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> res_seq(img.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(img.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_mpi);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_random_grayscale_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> img = gnitienko_k_functions::getRandomVector(151);
  double contrast_factor = 1.5;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> res_seq(img.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(img.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_mpi);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_incorrect_grayscale_image) {
  boost::mpi::communicator world;
  std::vector<int> img = {1000, -255, 185, -45, 255};
  double contrast_factor = 1.5;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> res_seq(img.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(img.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_mpi);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_color_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> img = {120, 240, 185, 100, 255, 0};
  double contrast_factor = 1.5;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> res_seq(img.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(img.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_mpi);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_random_color_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> img = gnitienko_k_functions::getRandomVector(150);
  double contrast_factor = 2.0;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> res_seq(img.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(img.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_mpi);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_incorrect_color_image) {
  boost::mpi::communicator world;
  std::vector<int> img = {1000, -255, 185, -45, 255, 0};
  double contrast_factor = 2.7;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> res_seq(img.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(img.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_mpi);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_empty_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> img = {};
  double contrast_factor = 1.5;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> res_seq(img.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(img.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res_mpi);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_grayscale_pixel_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> img = {123, 220};
  double contrast_factor = 3.5;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(110, res_mpi[0]);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, Test_negative_contrast_factor) {
  boost::mpi::communicator world;
  std::vector<uint8_t> img = {123, 220};
  double contrast_factor = -1.5;
  std::vector<uint8_t> res_mpi(img.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(img.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_mpi.data()));
    taskDataPar->outputs_count.emplace_back(res_mpi.size());
  }

  gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0)
    ASSERT_FALSE(testMpiTaskParallel.validation());
  else
    ASSERT_TRUE(true);
}