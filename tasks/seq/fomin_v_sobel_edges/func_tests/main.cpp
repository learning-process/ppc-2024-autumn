#include <gtest/gtest.h>

#include <vector>

#include "seq/fomin_v_sobel_edges/include/ops_seq.hpp"

TEST(Sequential_Sobel_Edge_Detection, Test_Sobel_4x4) {
  // Создание тестового изображения 4x4
  const int width = 4;
  const int height = 4;
  std::vector<unsigned char> input_image = {100, 100, 100, 100, 100, 200, 200, 100,
                                            100, 200, 200, 100, 100, 100, 100, 100};
  std::vector<unsigned char> output_image(width * height, 0);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Создание и выполнение задачи
  fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(taskDataSeq);
  ASSERT_EQ(sobelEdgeDetection.validation(), true);
  sobelEdgeDetection.pre_processing();
  sobelEdgeDetection.run();
  sobelEdgeDetection.post_processing();

  // Проверка, что выходное изображение не пустое
  bool is_output_valid = false;
  for (const auto& pixel : output_image) {
    if (pixel != 0) {
      is_output_valid = true;
      break;
    }
  }
  ASSERT_TRUE(is_output_valid);
}

TEST(Sequential_Sobel_Edge_Detection, Test_Sobel_8x8) {
  const int width = 8;
  const int height = 8;
  std::vector<unsigned char> input_image(width * height, 100);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      input_image[i * width + j] = 200;
    }
  }
  std::vector<unsigned char> output_image(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(taskDataSeq);
  ASSERT_EQ(sobelEdgeDetection.validation(), true);
  sobelEdgeDetection.pre_processing();
  sobelEdgeDetection.run();
  sobelEdgeDetection.post_processing();

  bool is_output_valid = false;
  for (const auto& pixel : output_image) {
    if (pixel != 0) {
      is_output_valid = true;
      break;
    }
  }
  ASSERT_TRUE(is_output_valid);
}

TEST(Sequential_Sobel_Edge_Detection, Test_Sobel_16x16) {
  const int width = 16;
  const int height = 16;
  std::vector<unsigned char> input_image(width * height, 100);
  for (int i = 4; i < 12; ++i) {
    for (int j = 4; j < 12; ++j) {
      input_image[i * width + j] = 200;
    }
  }
  std::vector<unsigned char> output_image(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(taskDataSeq);
  ASSERT_EQ(sobelEdgeDetection.validation(), true);
  sobelEdgeDetection.pre_processing();
  sobelEdgeDetection.run();
  sobelEdgeDetection.post_processing();

  bool is_output_valid = false;
  for (const auto& pixel : output_image) {
    if (pixel != 0) {
      is_output_valid = true;
      break;
    }
  }
  ASSERT_TRUE(is_output_valid);
}

TEST(Sequential_Sobel_Edge_Detection, Test_Sobel_32x32) {
  const int width = 32;
  const int height = 32;
  std::vector<unsigned char> input_image(width * height, 100);
  for (int i = 8; i < 24; ++i) {
    for (int j = 8; j < 24; ++j) {
      input_image[i * width + j] = 200;
    }
  }
  std::vector<unsigned char> output_image(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(taskDataSeq);
  ASSERT_EQ(sobelEdgeDetection.validation(), true);
  sobelEdgeDetection.pre_processing();
  sobelEdgeDetection.run();
  sobelEdgeDetection.post_processing();

  bool is_output_valid = false;
  for (const auto& pixel : output_image) {
    if (pixel != 0) {
      is_output_valid = true;
      break;
    }
  }
  ASSERT_TRUE(is_output_valid);
}

TEST(Sequential_Sobel_Edge_Detection, Test_Sobel_64x64) {
  const int width = 64;
  const int height = 64;
  std::vector<unsigned char> input_image(width * height, 100);
  for (int i = 16; i < 48; ++i) {
    for (int j = 16; j < 48; ++j) {
      input_image[i * width + j] = 200;
    }
  }
  std::vector<unsigned char> output_image(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  fomin_v_sobel_edges::SobelEdgeDetection sobelEdgeDetection(taskDataSeq);
  ASSERT_EQ(sobelEdgeDetection.validation(), true);
  sobelEdgeDetection.pre_processing();
  sobelEdgeDetection.run();
  sobelEdgeDetection.post_processing();

  bool is_output_valid = false;
  for (const auto& pixel : output_image) {
    if (pixel != 0) {
      is_output_valid = true;
      break;
    }
  }
  ASSERT_TRUE(is_output_valid);
}
