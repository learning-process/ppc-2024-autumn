// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kolodkin_g_image_contrast/include/ops_mpi.hpp"

TEST(kolodkin_g_image_contrast_MPI, Test_validation) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image.push_back(50);
    image.push_back(14);
    image.push_back(5);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
  }
}

TEST(kolodkin_g_image_contrast_MPI, Test_image_preprocessing) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image.push_back(50);
    image.push_back(14);
    image.push_back(5);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
  }
}

TEST(kolodkin_g_image_contrast_MPI, Test_run) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image.push_back(50);
    image.push_back(14);
    image.push_back(5);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
  }
}

TEST(kolodkin_g_image_contrast_MPI, Test_postprocessing) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image.push_back(50);
    image.push_back(14);
    image.push_back(5);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
  }
}

TEST(kolodkin_g_image_contrast_MPI, Test_global_and_reference) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image.push_back(50);
    image.push_back(14);
    image.push_back(5);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
  }
}

TEST(kolodkin_g_image_contrast_MPI, Test_image_one_pixel) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);
  std::vector<int> result_out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image.push_back(50);
    image.push_back(14);
    image.push_back(5);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  result_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);
    std::vector<int> result2_out;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    result2_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned long i = 0; i < result_out.size(); i++) {
      ASSERT_EQ(result_out[i], result2_out[i]);
    }
  }
}

TEST(kolodkin_g_image_contrast_MPI, Test_incorrect_image) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = {50, 14};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), false);
  }
}
TEST(kolodkin_g_image_contrast_MPI, Test_image_two_pixels) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(6, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image.push_back(50);
    image.push_back(14);
    image.push_back(5);
    image.push_back(10);
    image.push_back(200);
    image.push_back(105);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(6, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned long i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
  }
}
TEST(kolodkin_g_image_contrast_MPI, Test_incorrect_color_image) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = {50, 14, 256};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(3, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), false);
  }
}
TEST(kolodkin_g_image_contrast_MPI, Test_big_image) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(999, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (unsigned long i = 0; i < 999; i++) {
      image.push_back(0 + rand() % 255);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(999, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

    // Create Task
    kolodkin_g_image_contrast_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned long i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
  }
}