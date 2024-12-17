#include "seq/koshkin_n_linear_histogram_stretch/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

std::vector<int> koshkin_n_linear_histogram_stretch_seq::getRandomImage(int sz) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

bool koshkin_n_linear_histogram_stretch_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  image_input = std::vector<int>(size);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + size, image_input.begin());
  image_output = {};
  return true;
}

bool koshkin_n_linear_histogram_stretch_seq::TestTaskSequential::validation() {
  internal_order_test();
  int size = taskData->inputs_count[0];
  if (size % 3 != 0) return false;

  for (int i = 0; i < size; ++i) {
    int value = reinterpret_cast<int*>(taskData->inputs[0])[i];
    if (value < 0 || value > 255) {
      return false;
    }
  }

  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (!taskData->inputs_count.empty() && taskData->inputs_count[0] != 0) &&
          (!taskData->outputs_count.empty() && taskData->outputs_count[0] != 0));
}

bool koshkin_n_linear_histogram_stretch_seq::TestTaskSequential::run() {
  internal_order_test();
  // На вход принимается линейная матрица пикселей цветного (на деле это вектор)
  // "Линеаризованная матрица в виде: [R1, G1, B1, R2, G2, B2, ..., RN, GN, BN]"
  // Просто предположим, что изображение уже проанализировано по высоте и ширине
  // и значения каждого цвета пикселя передано в вектор вида выше
  int size = image_input.size();
  image_output.resize(size);
  int Imin = 255, Imax = 0;  // Минимальная яркость / Максимальная яркость

  // Перевод RGB в яркость (luminance) по стандартной формуле I=0.299⋅R+0.587⋅G+0.114⋅B
  // Расчёт яркостей и нахождение Imin, Imax
  std::vector<int> I(size / 3);  // Массив яркостей
  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int R = image_input[i];
    int G = image_input[i + 1];
    int B = image_input[i + 2];

    // Вычисление яркости
    I[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);

    if (I[k] < Imin) Imin = I[k];
    if (I[k] > Imax) Imax = I[k];
  }

  // Проверка, чтобы избежать деления на ноль
  if (Imin == Imax) {
    // Если гистограмма плоская, контраст не изменяется, возвращаем исходное изображение
    image_output = image_input;
    return true;
  }

  // Линейная растяжка яркости и восстановление изображения
  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    // Линейная растяжка (Рассчёт новых значений яркостей)
    int Inew = ((I[k] - Imin) * 255) / (Imax - Imin);

    // Коэффициент масштабирования
    float coeff = static_cast<float>(Inew) / static_cast<float>(I[k]);

    // Обновление каналов RGB
    image_output[i] = std::min(255, static_cast<int>(image_input[i] * coeff));
    image_output[i + 1] = std::min(255, static_cast<int>(image_input[i + 1] * coeff));
    image_output[i + 2] = std::min(255, static_cast<int>(image_input[i + 2] * coeff));
  }

  return true;
}

bool koshkin_n_linear_histogram_stretch_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(image_output.begin(), image_output.end(), output);
  return true;
}