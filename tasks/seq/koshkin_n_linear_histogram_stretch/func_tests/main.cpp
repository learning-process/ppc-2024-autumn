#include <gtest/gtest.h>

#include <vector>

#include "seq/koshkin_n_linear_histogram_stretch/include/ops_seq.hpp"

TEST(koshkin_n_linear_histogram_stretch_seq, test_correct_image) {
  const int count_size_vector = 6;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {15, 50, 45, 101, 92, 79};
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out = {0, 0, 0, 255, 252, 216};

  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(res_exp_out, out_vec);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_correct_image2) {
  const int width = 20;
  const int height = 20;
  const int count_size_vector = width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> in_vec = koshkin_n_linear_histogram_stretch_seq::getRandomImage(count_size_vector);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out(count_size_vector, 0);

  int Imin = 255;
  int Imax = 0;
  std::vector<int> intensity(width * height);
  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    int R = in_vec[i];
    int G = in_vec[i + 1];
    int B = in_vec[i + 2];
    intensity[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);
    Imin = std::min(Imin, intensity[k]);
    Imax = std::max(Imax, intensity[k]);
  }

  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    if (Imin == Imax) {
      res_exp_out[i] = in_vec[i];
      res_exp_out[i + 1] = in_vec[i + 1];
      res_exp_out[i + 2] = in_vec[i + 2];
      continue;
    }
    int Inew = ((intensity[k] - Imin) * 255) / (Imax - Imin);
    float coeff = static_cast<float>(Inew) / static_cast<float>(intensity[k]);
    res_exp_out[i] = std::min(255, static_cast<int>(in_vec[i] * coeff));
    res_exp_out[i + 1] = std::min(255, static_cast<int>(in_vec[i + 1] * coeff));
    res_exp_out[i + 2] = std::min(255, static_cast<int>(in_vec[i + 2] * coeff));
  }

  // Создаем TaskData
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Выполняем тест
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Проверка результата
  ASSERT_EQ(res_exp_out, out_vec);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_correct_image3) {
  const int width = 128;
  const int height = 128;
  const int count_size_vector = width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> in_vec = koshkin_n_linear_histogram_stretch_seq::getRandomImage(count_size_vector);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out(count_size_vector, 0);

  int Imin = 255;
  int Imax = 0;
  std::vector<int> intensity(width * height);
  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    int R = in_vec[i];
    int G = in_vec[i + 1];
    int B = in_vec[i + 2];
    intensity[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);
    Imin = std::min(Imin, intensity[k]);
    Imax = std::max(Imax, intensity[k]);
  }

  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    if (Imin == Imax) {
      res_exp_out[i] = in_vec[i];
      res_exp_out[i + 1] = in_vec[i + 1];
      res_exp_out[i + 2] = in_vec[i + 2];
      continue;
    }
    int Inew = ((intensity[k] - Imin) * 255) / (Imax - Imin);
    float coeff = static_cast<float>(Inew) / static_cast<float>(intensity[k]);
    res_exp_out[i] = std::min(255, static_cast<int>(in_vec[i] * coeff));
    res_exp_out[i + 1] = std::min(255, static_cast<int>(in_vec[i + 1] * coeff));
    res_exp_out[i + 2] = std::min(255, static_cast<int>(in_vec[i + 2] * coeff));
  }

  // Создаем TaskData
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Выполняем тест
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Проверка результата
  ASSERT_EQ(res_exp_out, out_vec);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_correct_image4) {
  const int width = 1024;
  const int height = 512;
  const int count_size_vector = width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<int> in_vec = koshkin_n_linear_histogram_stretch_seq::getRandomImage(count_size_vector);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out(count_size_vector, 0);

  int Imin = 255;
  int Imax = 0;
  std::vector<int> intensity(width * height);
  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    int R = in_vec[i];
    int G = in_vec[i + 1];
    int B = in_vec[i + 2];
    intensity[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);
    Imin = std::min(Imin, intensity[k]);
    Imax = std::max(Imax, intensity[k]);
  }

  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    if (Imin == Imax) {
      res_exp_out[i] = in_vec[i];
      res_exp_out[i + 1] = in_vec[i + 1];
      res_exp_out[i + 2] = in_vec[i + 2];
      continue;
    }
    int Inew = ((intensity[k] - Imin) * 255) / (Imax - Imin);
    float coeff = static_cast<float>(Inew) / static_cast<float>(intensity[k]);
    res_exp_out[i] = std::min(255, static_cast<int>(in_vec[i] * coeff));
    res_exp_out[i + 1] = std::min(255, static_cast<int>(in_vec[i + 1] * coeff));
    res_exp_out[i + 2] = std::min(255, static_cast<int>(in_vec[i + 2] * coeff));
  }

  // Создаем TaskData
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Выполняем тест
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Проверка результата
  ASSERT_EQ(res_exp_out, out_vec);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_incorrect_rgb_size_image) {
  const int count_size_vector = 8;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {15, 50, 45, 101, 92, 79, 0, 0};
  std::vector<int> out_vec(count_size_vector, 0);

  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_NE(testTaskSequential.validation(), true);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_incorrect_value_color_range_image) {
  const int count_size_vector = 12;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {631, -2, 45, 101, 92, 79, 0, 0, 300, 255, 10, 15};
  std::vector<int> out_vec(count_size_vector, 0);

  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_empty_image) {
  const int count_size_vector = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  koshkin_n_linear_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  // Create data
  std::vector<int> in_vec = {};
  std::vector<int> out_vec(count_size_vector, 0);

  // Create TaskData

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

  // Create Task
  ASSERT_EQ(testTaskSequential.validation(), false);
}