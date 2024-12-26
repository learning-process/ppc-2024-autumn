#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

TEST(mpi_sobel_edge_detection_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  // Создание тестового изображения (например, 8x8)
  const int width = 8;
  const int height = 8;
  global_image.resize(width * height, 100);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      global_image[i * width + j] = 200;
    }
  }

  // Создание TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_output_image.resize(width * height, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_output_image.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  // Создание и выполнение параллельной задачи
  auto sobelEdgeDetectionMPI = std::make_shared<fomin_v_sobel_edges::SobelEdgeDetectionMPI>(taskDataPar);
  ASSERT_EQ(sobelEdgeDetectionMPI->validation(), true);
  sobelEdgeDetectionMPI->pre_processing();
  sobelEdgeDetectionMPI->run();
  sobelEdgeDetectionMPI->post_processing();

  // Создание Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeDetectionMPI);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка, что выходное изображение не пустое
    bool is_output_valid = false;
    for (const auto& pixel : global_output_image) {
      if (pixel != 0) {
        is_output_valid = true;
        break;
      }
    }
    ASSERT_TRUE(is_output_valid);
  }
}

TEST(mpi_sobel_edge_detection_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<unsigned char> global_image;
  std::vector<unsigned char> global_output_image;

  // Создание тестового изображения
  const int width = 8;
  const int height = 8;
  global_image.resize(width * height, 100);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      global_image[i * width + j] = 200;
    }
  }

  // Создание TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_output_image.resize(width * height, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_image.data()));
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_output_image.data()));
    taskDataPar->outputs_count.emplace_back(width);
    taskDataPar->outputs_count.emplace_back(height);
  }

  // Создание и выполнение параллельной задачи
  auto sobelEdgeDetectionMPI = std::make_shared<fomin_v_sobel_edges::SobelEdgeDetectionMPI>(taskDataPar);
  ASSERT_EQ(sobelEdgeDetectionMPI->validation(), true);
  sobelEdgeDetectionMPI->pre_processing();
  sobelEdgeDetectionMPI->run();
  sobelEdgeDetectionMPI->post_processing();

  // Создание Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelEdgeDetectionMPI);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка, что выходное изображение не пустое
    bool is_output_valid = false;
    for (const auto& pixel : global_output_image) {
      if (pixel != 0) {
        is_output_valid = true;
        break;
      }
    }
    ASSERT_TRUE(is_output_valid);
  }
}