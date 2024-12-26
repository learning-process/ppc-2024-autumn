#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/oturin_a_histogram_stretch/include/ops_mpi.hpp"

namespace oturin_a_histogram_stretch_mpi {
std::vector<uint8_t> getRandomGrayscaleImg(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<uint8_t> vec(sz * 3);
  for (size_t i = 0; i < vec.size(); i += 3) {
    vec[i] = gen() % 256;
    vec[i + 1] = vec[i];
    vec[i + 2] = vec[i];
  }
  return vec;
}

}  // namespace oturin_a_histogram_stretch_mpi

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_1px) {
  boost::mpi::communicator world;

  int width = 1;
  int height = 1;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_2px) {
  boost::mpi::communicator world;

  int width = 2;
  int height = 1;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_3px) {
  boost::mpi::communicator world;

  int width = 1;
  int height = 3;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_4px) {
  boost::mpi::communicator world;

  int width = 2;
  int height = 2;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_5px) {
  boost::mpi::communicator world;

  int width = 5;
  int height = 1;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_square) {
  boost::mpi::communicator world;

  int width = 20;
  int height = 20;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_landscape) {
  boost::mpi::communicator world;

  int width = 15;
  int height = 5;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_RANDOM_portrait) {
  boost::mpi::communicator world;

  int width = 5;
  int height = 15;

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create data
    startImage = oturin_a_histogram_stretch_mpi::getRandomGrayscaleImg(width * height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_LINE) {
  boost::mpi::communicator world;

  int width{};
  int height{};

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
    std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
    std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

    std::string filenameOriginal = dir_path + "/../line.bmp";

    // Create data
    startImage = oturin_a_histogram_stretch_mpi::ReadBMP(filenameOriginal.c_str(), width, height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_PAYSAGE) {
  boost::mpi::communicator world;

  int width{};
  int height{};

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
    std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
    std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

    std::string filenameOriginal = dir_path + "/../paysage.bmp";

    // Create data
    startImage = oturin_a_histogram_stretch_mpi::ReadBMP(filenameOriginal.c_str(), width, height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}

TEST(oturin_a_histogram_stretch_mpi_functest, Test_IMAGE_SQUARE) {
  boost::mpi::communicator world;

  int width{};
  int height{};

  std::vector<uint8_t> startImage;
  std::vector<uint8_t> parallelResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
    std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
    std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

    std::string filenameOriginal = dir_path + "/../square.bmp";

    // Create data
    startImage = oturin_a_histogram_stretch_mpi::ReadBMP(filenameOriginal.c_str(), width, height);
    parallelResult = std::vector<uint8_t>(width * height * 3);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(startImage.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallelResult.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  oturin_a_histogram_stretch_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    oturin_a_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallelResult, seqResult);
  }
}
