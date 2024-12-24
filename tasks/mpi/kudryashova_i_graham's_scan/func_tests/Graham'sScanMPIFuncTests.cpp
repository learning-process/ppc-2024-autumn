#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <numbers>

#include "mpi/kudryashova_i_graham's_scan/include/Graham'sScanMPI.hpp"

void generateUniquePoints(int numPoints, int8_t minX, int8_t maxX, int8_t minY, int8_t maxY,
                          std::vector<int8_t> &xCoords, std::vector<int8_t> &yCoords) {
  if (numPoints > (maxX - minX + 1) * (maxY - minY + 1)) {
    std::cerr << "Error: Not enough unique points can be generated in the given range." << std::endl;
    return;
  }
  std::vector<std::pair<int8_t, int8_t>> allPoints;
  for (int8_t x = minX; x <= maxX; x += 1) {
    for (int8_t y = minY; y <= maxY; y += 1) {
      allPoints.emplace_back(x, y);
    }
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(allPoints.begin(), allPoints.end(), gen);
  for (int i = 0; i < numPoints; ++i) {
    xCoords.push_back(allPoints[i].first);
    yCoords.push_back(allPoints[i].second);
  }
}

void generateUniquePointsCircle(int numPoints, int8_t centerX, int8_t centerY, int radius, std::vector<int8_t> &xCoords,
                                std::vector<int8_t> &yCoords) {
  if (numPoints > 2 * std::numbers::pi * radius) {
    std::cerr << "Error: Not enough unique points can be generated on the circle's surface." << std::endl;
    return;
  }
  std::vector<std::pair<int8_t, int8_t>> allPoints;
  for (int i = 0; i < numPoints; ++i) {
    double angle = static_cast<double>(i) / numPoints * 2 * std::numbers::pi;
    auto x = static_cast<int8_t>(centerX + radius * cos(angle));
    auto y = static_cast<int8_t>(centerY + radius * sin(angle));
    allPoints.emplace_back(x, y);
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(allPoints.begin(), allPoints.end(), gen);
  for (const auto &point : allPoints) {
    xCoords.push_back(point.first);
    yCoords.push_back(point.second);
  }
}

void generateUniquePointsRhombus(int numPoints, int centerX, int centerY, int d1, int d2, std::vector<int8_t> &xCoords,
                                 std::vector<int8_t> &yCoords) {
  if (numPoints > 2 * (d1 + d2)) {
    std::cerr << "Error: Not enough unique points can be generated on the diamond's surface." << std::endl;
    return;
  }
  std::vector<std::pair<int8_t, int8_t>> allPoints;
  for (int i = 0; i < numPoints; ++i) {
    double ratio = static_cast<double>(i) / numPoints;
    if (ratio < 0.25) {
      double x = centerX + (d1 / 2) * (ratio / 0.25);
      double y = centerY + (d2 / 2);
      allPoints.emplace_back(static_cast<int>(x), static_cast<int>(y));
    } else if (ratio < 0.5) {
      double x = centerX + (d1 / 2);
      double y = centerY - (d2 / 2) + (d2 * (ratio - 0.25) / 0.25);
      allPoints.emplace_back(static_cast<int>(x), static_cast<int>(y));
    } else if (ratio < 0.75) {
      double x = centerX - (d1 / 2) * ((ratio - 0.5) / 0.25);
      double y = centerY - (d2 / 2);
      allPoints.emplace_back(static_cast<int>(x), static_cast<int>(y));
    } else {
      double x = centerX - (d1 / 2);
      double y = centerY + (d2 / 2) - (d2 * (ratio - 0.75) / 0.25);
      allPoints.emplace_back(static_cast<int>(x), static_cast<int>(y));
    }
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(allPoints.begin(), allPoints.end(), gen);
  for (const auto &point : allPoints) {
    xCoords.push_back(point.first);
    yCoords.push_back(point.second);
  }
}

void addAns(std::vector<int8_t> &v1, std::vector<int8_t> &v2, int value) {
  v1.push_back(-value);
  v2.push_back(-value);
  v1.push_back(value);
  v2.push_back(-value);
  v1.push_back(value);
  v2.push_back(value);
  v1.push_back(-value);
  v2.push_back(value);
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_random_test_rhombus) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  const int count_size = 500;
  const int d1 = 150;
  const int d2 = 100;
  const int centre_x = 0;
  const int centre_y = 0;
  std::vector<int8_t> result(count_size * 2, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x;
    std::vector<int8_t> vector_y;
    generateUniquePointsRhombus(count_size, centre_x, centre_y, d1, d2, vector_x, vector_y);
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(count_size * 2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_random_test_circle) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  const int count_size = 500;
  const int radius = 80;
  const int centre_x = 0;
  const int centre_y = 0;
  std::vector<int8_t> result(count_size * 2, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x;
    std::vector<int8_t> vector_y;
    generateUniquePointsCircle(count_size, centre_x, centre_y, radius, vector_x, vector_y);
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(count_size * 2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_random_test_square) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  const int count_size = 800;
  const int width = 15;
  const int length = 15;
  std::vector<int8_t> result(count_size * 2, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x;
    std::vector<int8_t> vector_y;
    generateUniquePoints(count_size, -width, width, -length, length, vector_x, vector_y);
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(count_size * 2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_random_test_rectangle) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  const int count_size = 1000;
  const int width = 15;
  const int length = 20;
  std::vector<int8_t> result(count_size * 2, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x;
    std::vector<int8_t> vector_y;
    generateUniquePoints(count_size, -width, width, -length, length, vector_x, vector_y);
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(count_size * 2, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_test_square) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(8, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x = {0, 0, 1, 1};
    std::vector<int8_t> vector_y = {1, 0, 0, 1};
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(8, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_test_star) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(10, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x = {0, 1, 4, 2, 3, 0, -3, -2, -4, -1};
    std::vector<int8_t> vector_y = {4, 1, 1, -1, -4, -2, -4, -1, 1, 1};
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(10, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_simple_test_1) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(8, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x = {3, 0, 2, 3, 1, -1, 1, 0, 3, -3, -3};
    std::vector<int8_t> vector_y = {5, 3, 2, 2, 1, 1, 0, 0, -1, 2, -2};
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(8, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_simple_test_2) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(12, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x = {5, 3, 3, 1, 2, 4, 1, 1, 2, 1, -1, -2, -1, -1};
    std::vector<int8_t> vector_y = {3, 3, 2, 2, 1, -2, -1, -2, -3, -4, -3, -1, 1, 3};
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(12, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_random_test) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  const int count_size = 15;
  const int ans_number = 15;
  std::vector<int8_t> result(8, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x;
    std::vector<int8_t> vector_y;
    generateUniquePoints(count_size, -(ans_number - 1), (ans_number - 1), -(ans_number - 1), (ans_number - 1), vector_x,
                         vector_y);
    addAns(vector_x, vector_y, ans_number);
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(8, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_random_test_no_ans) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  const int count_size = 100;
  const int ans_number = 50;
  std::vector<int8_t> result(50, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x;
    std::vector<int8_t> vector_y;
    generateUniquePoints(count_size, -(ans_number - 1), (ans_number - 1), -(ans_number - 1), (ans_number - 1), vector_x,
                         vector_y);
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int8_t> reference(50, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataSeq->inputs_count.emplace_back(global_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(result, reference);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_check_same_number_x_and_y) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(8, 0);
  std::vector<int8_t> vector_x = {2, 0, -2, 0};
  std::vector<int8_t> vector_y = {0, 2, 0, -2};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_check_not_same_number_x_and_y) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(8, 0);
  std::vector<int8_t> vector_x = {2, 0, -2, 0, 2};
  std::vector<int8_t> vector_y = {0, 2, 0, -2};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_check_empty_vertex) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x;
  std::vector<int8_t> vector_y;
  std::vector<int8_t> result(6, 0);
  if (world.rank() == 0) {
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kudryashova_i_graham_scan_mpi, mpi_graham_scan_test_1_vertex) {
  boost::mpi::communicator world;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(6, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int8_t> vector_x = {1};
    std::vector<int8_t> vector_y = {1};
    global_vector.reserve(vector_x.size() + vector_y.size());
    global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
    global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
    kudryashova_i_graham_scan_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}