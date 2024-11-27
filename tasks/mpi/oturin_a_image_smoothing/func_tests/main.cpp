#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/oturin_a_image_smoothing/include/ops_mpi.hpp"

std::vector<uint8_t> oturin_a_image_smoothing_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<uint8_t> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

TEST(oturin_a_image_smoothing_mpi_functest, Test_IMAGE_RANDOM_square) {
  boost::mpi::communicator world;

  int width = 100;
  int height = 100;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_image_smoothing_mpi::getRandomVector(width * height * 3);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_image_smoothing_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> seqResult(width * height * 3);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.emplace_back(width);
    taskDataSeq->outputs_count.emplace_back(height);

    // Create Task
    oturin_a_image_smoothing_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_image_smoothing_mpi_functest, Test_IMAGE_RANDOM_landscape) {
  boost::mpi::communicator world;

  int width = 150;
  int height = 100;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_image_smoothing_mpi::getRandomVector(width * height * 3);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_image_smoothing_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> seqResult(width * height * 3);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.emplace_back(width);
    taskDataSeq->outputs_count.emplace_back(height);

    // Create Task
    oturin_a_image_smoothing_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_image_smoothing_mpi_functest, Test_IMAGE_RANDOM_portrait) {
  boost::mpi::communicator world;

  int width = 100;
  int height = 140;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_image_smoothing_mpi::getRandomVector(width * height * 3);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_image_smoothing_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> seqResult(width * height * 3);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.emplace_back(width);
    taskDataSeq->outputs_count.emplace_back(height);

    // Create Task
    oturin_a_image_smoothing_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_image_smoothing_mpi_functest, Test_IMAGE_LINE) {
  boost::mpi::communicator world;

  int width{};
  int height{};

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
    std::string dir_path = file_path.substr(0, file_path.rfind("\\"));
#else
    std::string dir_path = file_path.substr(0, file_path.rfind("/"));
#endif

    std::string filenameOriginal = dir_path + "/../line.bmp";

    // Create data
    startImage = oturin_a_image_smoothing_mpi::ReadBMP(filenameOriginal.c_str(), width, height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_image_smoothing_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> seqResult(width * height * 3);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.emplace_back(width);
    taskDataSeq->outputs_count.emplace_back(height);

    // Create Task
    oturin_a_image_smoothing_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_image_smoothing_mpi_functest, Test_IMAGE_CIRCLE) {
  boost::mpi::communicator world;

  int width{};
  int height{};

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
    std::string dir_path = file_path.substr(0, file_path.rfind("\\"));
#else
    std::string dir_path = file_path.substr(0, file_path.rfind("/"));
#endif

    std::string filenameOriginal = dir_path + "/../circle.bmp";

    // Create data
    startImage = oturin_a_image_smoothing_mpi::ReadBMP(filenameOriginal.c_str(), width, height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_image_smoothing_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> seqResult(width * height * 3);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.emplace_back(width);
    taskDataSeq->outputs_count.emplace_back(height);

    // Create Task
    oturin_a_image_smoothing_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_image_smoothing_mpi_functest, Test_IMAGE_COLOR) {
  boost::mpi::communicator world;

  int width{};
  int height{};

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
    std::string dir_path = file_path.substr(0, file_path.rfind("\\"));
#else
    std::string dir_path = file_path.substr(0, file_path.rfind("/"));
#endif

    std::string filenameOriginal = dir_path + "/../line.bmp";

    // Create data
    startImage = oturin_a_image_smoothing_mpi::ReadBMP(filenameOriginal.c_str(), width, height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_image_smoothing_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> seqResult(width * height * 3);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataSeq->inputs_count.emplace_back(width);
    taskDataSeq->inputs_count.emplace_back(height);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.emplace_back(width);
    taskDataSeq->outputs_count.emplace_back(height);

    // Create Task
    oturin_a_image_smoothing_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}
