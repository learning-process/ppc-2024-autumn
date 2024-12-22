// Filatev Vladislav Metod Belmana Forda
#include <gtest/gtest.h>

#include <vector>

#include "seq/filatev_v_metod_belmana_forda/include/ops_seq.hpp"

using namespace filatev_v_metod_belmana_forda_seq;

TEST(filatev_v_metod_belmana_forda_seq, test_simpel_path) {
  int n = 6;
  int m = 9;
  int start = 0;
  std::vector<int> Adjncy = {1, 2, 3, 4, 1, 4, 4, 5, 5};
  std::vector<int> Xadj = {0, 2, 4, 6, 8, 9, 9};
  std::vector<int> Eweights = {7, 9, -1, -2, -3, 2, 1, 3, 3};
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_TRUE(metodBelmanaForda.validation());
  metodBelmanaForda.pre_processing();
  metodBelmanaForda.run();
  metodBelmanaForda.post_processing();

  std::vector<int> tResh = {0, 6, 9, 5, 4, 7};

  ASSERT_EQ(tResh, d);
}

TEST(filatev_v_metod_belmana_forda_seq, test_simpel_path2) {
  int n = 4;
  int m = 12;
  int start = 0;
  std::vector<int> Adjncy = {1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2};
  std::vector<int> Xadj = {0, 3, 6, 9, 12, 12};
  std::vector<int> Eweights(m, 1);
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_TRUE(metodBelmanaForda.validation());
  metodBelmanaForda.pre_processing();
  metodBelmanaForda.run();
  metodBelmanaForda.post_processing();

  std::vector<int> tResh = {0, 1, 1, 1};

  ASSERT_EQ(tResh, d);
}

TEST(filatev_v_metod_belmana_forda_seq, test_simpel_path3) {
  int n = 7;
  int m = 12;
  int start = 0;
  std::vector<int> Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
  std::vector<int> Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
  std::vector<int> Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_TRUE(metodBelmanaForda.validation());
  metodBelmanaForda.pre_processing();
  metodBelmanaForda.run();
  metodBelmanaForda.post_processing();

  std::vector<int> tResh = {0, 7, 5, 7, 6, 8, 10};

  ASSERT_EQ(tResh, d);
}

TEST(filatev_v_metod_belmana_forda_seq, test_simpel_path4) {
  int inf = std::numeric_limits<int>::max();
  int n = 6;
  int m = 9;
  int start = 1;
  std::vector<int> Adjncy = {1, 2, 3, 4, 1, 4, 4, 5, 5};
  std::vector<int> Xadj = {0, 2, 4, 6, 8, 9, 9};
  std::vector<int> Eweights = {7, 9, -1, -2, -3, 2, 1, 3, 3};
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_TRUE(metodBelmanaForda.validation());
  metodBelmanaForda.pre_processing();
  metodBelmanaForda.run();
  metodBelmanaForda.post_processing();

  std::vector<int> tResh = {inf, 0, inf, -1, -2, 1};

  ASSERT_EQ(tResh, d);
}

TEST(filatev_v_metod_belmana_forda_seq, test_error) {
  int n = 7;
  int start = 0;
  std::vector<int> Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
  std::vector<int> Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
  std::vector<int> Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(0);
  taskData->inputs_count.emplace_back(0);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_FALSE(metodBelmanaForda.validation());
}

TEST(filatev_v_metod_belmana_forda_seq, test_error_2) {
  int n = 7;
  int start = 0;
  std::vector<int> Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
  std::vector<int> Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
  std::vector<int> Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(54);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_FALSE(metodBelmanaForda.validation());
}

TEST(filatev_v_metod_belmana_forda_seq, test_error_4) {
  int n = 7;
  int m = 12;
  std::vector<int> Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
  std::vector<int> Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
  std::vector<int> Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(25);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_FALSE(metodBelmanaForda.validation());
}

TEST(filatev_v_metod_belmana_forda_seq, test_error_3) {
  int n = 7;
  int m = 12;
  int start = 0;
  std::vector<int> Adjncy = {1, 2, 3, 6, 1, 4, 5, 4, 3, 0, 6, 0};
  std::vector<int> Xadj = {0, 2, 4, 7, 8, 9, 11, 12};
  std::vector<int> Eweights = {8, 5, 1, 8, 2, 1, 3, 2, 1, 5, 2, 4};
  std::vector<int> d(n);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n + 1);

  filatev_v_metod_belmana_forda_seq::MetodBelmanaForda metodBelmanaForda(taskData);

  ASSERT_FALSE(metodBelmanaForda.validation());
}