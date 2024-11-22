// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kolodkin_g_image_contrast/include/ops_mpi.hpp"

TEST(kolodkin_g_image_contrast_MPI, Test_image_one_pixel) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> image;
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    image = {50, 14, 5};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  } else {
    image.resize(3, 0);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(3, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataSeq->inputs_count.emplace_back(image.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(reference_out)));

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

TEST(kolodkin_g_image_contrast_MPI, Test_image_center_line) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> image;
  std::vector<int> global_out(300, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (unsigned long i = 0; i < 300; i++) {
      if (i >= 120 && i < 150) {
        image.push_back(255);
      } else {
        image.push_back(0);
      }
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  } else {
    image.resize(300, 0);
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
    std::vector<int> reference_out(300, 0);

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

TEST(kolodkin_g_image_contrast_MPI, Test_incorrect_image) {
  boost::mpi::communicator world;
  std::vector<int> image;

  // Create data
  std::vector<int> global_out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    image = {50, 14};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  } else {
    image.resize(2, 0);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  // Create Task
  kolodkin_g_image_contrast_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(2, 0);

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
    image = {50, 14, 5, 20, 10, 105};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataMpi->inputs_count.emplace_back(image.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  } else {
    image.resize(6, 0);
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
  } else {
    image.resize(3, 0);
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
  } else {
    image.resize(999, 0);
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