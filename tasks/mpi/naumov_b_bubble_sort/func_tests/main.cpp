#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "mpi/naumov_b_bubble_sort/include/ops_mpi.hpp"

TEST(naumov_b_bubble_sort_mpi, Test_10_int_with_logging) {
  const size_t length = 10;
  std::vector<int> in(length);
  std::vector<int> out(length);

  // Генерация случайных чисел
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(-1000, 1000);
  for (size_t i = 0; i < length; ++i) {
    in[i] = distribution(generator);
  }

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  // Логирование перед созданием объекта tmpPar
  if (world.rank() == 0) {
    std::cout << "Test_10_int_with_logging: preparing input data..." << std::endl;
    std::cout << "Input size: " << in.size() << std::endl;
    std::cout << "Input data (first 10 elements): ";
    for (int i = 0; i < 10 && i < in.size(); ++i) {
      std::cout << in[i] << " ";
    }
    std::cout << std::endl;
  }

  // Проверка, что указатели действительны
  if (in.empty() || out.empty()) {
    std::cerr << "Error: Input or output vectors are empty!" << std::endl;
  }

  // Заполнение данных в tmpPar
  if (world.rank() == 0) {
    try {
      tmpPar->inputs_count.emplace_back(in.size());
      tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));  // Здесь
      tmpPar->outputs_count.emplace_back(out.size());
    } catch (const std::exception &e) {
      std::cerr << "Error while initializing tmpPar: " << e.what() << std::endl;
    }
  }

  // Создание объекта для параллельной сортировки
  naumov_b_bubble_sort_mpi::TestMPITaskParallel tmpTaskPar(tmpPar);

  // Валидация данных перед обработкой
  if (world.rank() == 0) {
    std::cout << "Validation step..." << std::endl;
  }
  ASSERT_TRUE(tmpTaskPar.validation());  // Проверка валидности данных

  // Логирование после валидации
  if (world.rank() == 0) {
    std::cout << "Data validation passed." << std::endl;
  }

  // Преобразование и сортировка данных
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();

  // Логирование после завершения обработки
  if (world.rank() == 0) {
    std::cout << "Sorting completed. Checking results..." << std::endl;

    // Сравнение отсортированного массива с эталоном
    std::vector<int> reference = in;
    std::sort(reference.begin(), reference.end());
    ASSERT_EQ(out, reference);

    std::cout << "Test passed. Sorted array matches reference." << std::endl;
  }
}
