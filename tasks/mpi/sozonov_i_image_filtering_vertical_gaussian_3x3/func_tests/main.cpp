#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sozonov_i_image_filtering_vertical_gaussian_3x3/include/ops_mpi.hpp"

namespace sozonov_i_image_filtering_vertical_gaussian_3x3_mpi {

std::vector<double> getRandomImage(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(0, 255);
  std::vector<double> img(sz);
  for (int i = 0; i < sz; ++i) {
    img[i] = dis(gen);
  }
  return img;
}

}  // namespace sozonov_i_image_filtering_vertical_gaussian_3x3_mpi

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_image_less_than_3x3) {
  boost::mpi::communicator world;

  const int width = 2;
  const int height = 2;

  std::vector<double> global_img = {4, 6, 8, 24};
  std::vector<double> global_ans(width * height, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_wrong_pixels) {
  boost::mpi::communicator world;

  const int width = 5;
  const int height = 3;

  std::vector<double> global_img = {143, 6, 853, -24, 31, 25, 1, -5, 7, 361, -28, 98, -45, 982, 461};
  std::vector<double> global_ans(width * height, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_7x5) {
  boost::mpi::communicator world;

  const int width = 7;
  const int height = 5;

  std::vector<double> global_img = {34, 24, 27, 1,   67, 42, 48, 93, 26, 47, 2,  16, 34, 13, 81, 5, 24, 32,
                                    12, 2,  37, 123, 12, 78, 64, 1,  76, 7,  46, 9,  4,  10, 17, 4, 112};
  std::vector<double> global_ans(width * height, 0);
  std::vector<double> ans = {0,      0,       0,       0,      0,       0,     0,       42.6875, 38,
                             25.5,   20.625,  23.1875, 27.875, 20.875,  50.25, 40.4375, 32.75,   29.625,
                             20.375, 22.6875, 18.875,  49,     39.5625, 36,    34.6875, 24.375,  31.875,
                             30.25,  0,       0,       0,      0,       0,     0,       0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_ans, ans);
    ASSERT_EQ(global_ans, ans);
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_10x3) {
  boost::mpi::communicator world;

  const int width = 10;
  const int height = 3;

  std::vector<double> global_img = {34, 24, 27, 67,  42, 48, 93, 26, 47, 2,  34, 13, 81, 24, 32,
                                    12, 2,  37, 123, 12, 78, 64, 1,  7,  46, 9,  10, 17, 4,  112};
  std::vector<double> global_ans(width * height, 0);
  std::vector<double> ans = {0,      0,      0,    0,      0,       0,       0,     0,      0,       0,
                             29.625, 37.375, 38.5, 36.625, 31.6875, 26.3125, 25.75, 39.875, 53.0625, 35.8125,
                             0,      0,      0,    0,      0,       0,       0,     0,      0,       0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_ans, ans);
    ASSERT_EQ(global_ans, ans);
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_100x100) {
  boost::mpi::communicator world;

  const int width = 100;
  const int height = 100;

  std::vector<double> global_img(width * height, 1);
  std::vector<double> global_ans(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_ans, reference_ans);
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_150x200) {
  boost::mpi::communicator world;

  const int width = 150;
  const int height = 200;

  std::vector<double> global_img(width * height, 1);
  std::vector<double> global_ans(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_ans, reference_ans);
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_random_100x100) {
  boost::mpi::communicator world;

  const int width = 100;
  const int height = 100;

  std::vector<double> global_img(width * height);
  std::vector<double> global_ans(width * height, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_img = sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::getRandomImage(width * height);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_ans, reference_ans);
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_random_150x100) {
  boost::mpi::communicator world;

  const int width = 150;
  const int height = 100;

  std::vector<double> global_img(width * height);
  std::vector<double> global_ans(width * height, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_img = sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::getRandomImage(width * height);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_ans, reference_ans);
  }
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_mpi, test_random_120x200) {
  boost::mpi::communicator world;

  const int width = 120;
  const int height = 200;

  std::vector<double> global_img(width * height);
  std::vector<double> global_ans(width * height, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_img = sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::getRandomImage(width * height);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(width * height, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataSeq->inputs_count.emplace_back(global_img.size());
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_image_filtering_vertical_gaussian_3x3_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_ans, reference_ans);
  }
}