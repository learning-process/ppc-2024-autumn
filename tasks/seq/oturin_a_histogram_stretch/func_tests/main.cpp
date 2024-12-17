#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>
#include <vector>

#include "seq/oturin_a_histogram_stretch/include/ops_seq.hpp"

TEST(oturin_a_histogram_stretch_functest, Test_IMAGE_LINE) {
  std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
  std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
  std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

  std::string filenameOriginal = dir_path + "/../line.bmp";
  std::string filenameCompare = dir_path + "/../lineREF.bmp";

  int width{};
  int height{};

  // Create data
  std::vector<uint8_t> in = oturin_a_histogram_stretch_seq::ReadBMP(filenameOriginal.c_str(), width, height);
  std::vector<uint8_t> ref = oturin_a_histogram_stretch_seq::ReadBMP(filenameCompare.c_str(), width, height);
  std::vector<uint8_t> out(width * height * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true) << filenameOriginal;
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}

TEST(oturin_a_histogram_stretch_functest, Test_IMAGE_MAN) {
  std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
  std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
  std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

  std::string filenameOriginal = dir_path + "/../man.bmp";
  std::string filenameCompare = dir_path + "/../manREF.bmp";

  int width{};
  int height{};

  // Create data
  std::vector<uint8_t> in = oturin_a_histogram_stretch_seq::ReadBMP(filenameOriginal.c_str(), width, height);
  std::vector<uint8_t> ref = oturin_a_histogram_stretch_seq::ReadBMP(filenameCompare.c_str(), width, height);
  std::vector<uint8_t> out(width * height * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true) << filenameOriginal;
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}

TEST(oturin_a_histogram_stretch_functest, Test_IMAGE_PAYSAGE) {
  std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
  std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
  std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

  std::string filenameOriginal = dir_path + "/../paysage.bmp";
  std::string filenameCompare = dir_path + "/../paysageREF.bmp";

  int width{};
  int height{};

  // Create data
  std::vector<uint8_t> in = oturin_a_histogram_stretch_seq::ReadBMP(filenameOriginal.c_str(), width, height);
  std::vector<uint8_t> ref = oturin_a_histogram_stretch_seq::ReadBMP(filenameCompare.c_str(), width, height);
  std::vector<uint8_t> out(width * height * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true) << filenameOriginal;
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}

TEST(oturin_a_histogram_stretch_functest, Test_IMAGE_CODE) {
  std::string file_path = __FILE__;
#if defined(_WIN32) || defined(WIN32)
  std::string dir_path = file_path.substr(0, file_path.rfind('\\'));
#else
  std::string dir_path = file_path.substr(0, file_path.rfind('/'));
#endif

  std::string filenameOriginal = dir_path + "/../code.bmp";
  std::string filenameCompare = dir_path + "/../codeREF.bmp";

  int width{};
  int height{};

  // Create data
  std::vector<uint8_t> in = oturin_a_histogram_stretch_seq::ReadBMP(filenameOriginal.c_str(), width, height);
  std::vector<uint8_t> ref = oturin_a_histogram_stretch_seq::ReadBMP(filenameCompare.c_str(), width, height);
  std::vector<uint8_t> out(width * height * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true) << filenameOriginal;
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}

TEST(oturin_a_histogram_stretch_functest, Test_IMAGE_ONECOLOR) {
  int width = 10;
  int height = 10;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 0);
  std::vector<uint8_t> ref(width * height * 3, 0);
  std::vector<uint8_t> out(width * height * 3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  oturin_a_histogram_stretch_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, out);
}