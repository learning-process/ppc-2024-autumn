#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

TEST(shuravina_o_contrast, Test_Contrast_MinMax) {
  boost::mpi::communicator world;
  const int count = 10;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  if (world.rank() == 0) {
    for (int i = 0; i < count; ++i) {
      in[i] = (i % 2 == 0) ? 0 : 255;
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  shuravina_o_contrast::ContrastTaskParallel contrastTaskParallel(taskDataPar);
  ASSERT_EQ(contrastTaskParallel.validation(), true);
  contrastTaskParallel.pre_processing();
  contrastTaskParallel.run();
  contrastTaskParallel.post_processing();

  if (world.rank() == 0) {
    for (int i = 0; i < count; ++i) {
      ASSERT_EQ(out[i], (i % 2 == 0) ? 0 : 255);
    }
  }
}