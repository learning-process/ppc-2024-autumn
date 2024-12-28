// Copyright 2024 Anikin Maksim
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/anikin_m_contrastscale/include/ops_mpi.hpp"

using namespace anikin_m_contrastscale_mpi;

TEST(anikin_m_contrastscale_mpi, increase_contrast_10) {
  const int count = 10;
  float k = 1.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  // Seq
  if (world.rank() == 0) {
    std::vector<RGB> seq_out;
    seq_out.resize(count);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    ContrastScaleSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < count; i++) {
      ASSERT_EQ(seq_out[i].R, out[i].R);
      ASSERT_EQ(seq_out[i].G, out[i].G);
      ASSERT_EQ(seq_out[i].B, out[i].B);
    }
  }
}

TEST(anikin_m_contrastscale_mpi, increase_contrast_100) {
  const int count = 100;
  float k = 1.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  // Seq
  if (world.rank() == 0) {
    std::vector<RGB> seq_out;
    seq_out.resize(count);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    ContrastScaleSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < count; i++) {
      ASSERT_EQ(seq_out[i].R, out[i].R);
      ASSERT_EQ(seq_out[i].G, out[i].G);
      ASSERT_EQ(seq_out[i].B, out[i].B);
    }
  }
}

TEST(anikin_m_contrastscale_mpi, increase_contrast_500) {
  const int count = 500;
  float k = 1.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  // Seq
  if (world.rank() == 0) {
    std::vector<RGB> seq_out;
    seq_out.resize(count);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    ContrastScaleSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < count; i++) {
      ASSERT_EQ(seq_out[i].R, out[i].R);
      ASSERT_EQ(seq_out[i].G, out[i].G);
      ASSERT_EQ(seq_out[i].B, out[i].B);
    }
  }
}

TEST(anikin_m_contrastscale_mpi, decrease_contrast_10) {
  const int count = 10;
  float k = 0.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  // Seq
  if (world.rank() == 0) {
    std::vector<RGB> seq_out;
    seq_out.resize(count);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    ContrastScaleSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < count; i++) {
      ASSERT_EQ(seq_out[i].R, out[i].R);
      ASSERT_EQ(seq_out[i].G, out[i].G);
      ASSERT_EQ(seq_out[i].B, out[i].B);
    }
  }
}

TEST(anikin_m_contrastscale_mpi, decrease_contrast_100) {
  const int count = 100;
  float k = 0.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  // Seq
  if (world.rank() == 0) {
    std::vector<RGB> seq_out;
    seq_out.resize(count);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    ContrastScaleSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < count; i++) {
      ASSERT_EQ(seq_out[i].R, out[i].R);
      ASSERT_EQ(seq_out[i].G, out[i].G);
      ASSERT_EQ(seq_out[i].B, out[i].B);
    }
  }
}

TEST(anikin_m_contrastscale_mpi, decrease_contrast_500) {
  const int count = 500;
  float k = 0.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  // Seq
  if (world.rank() == 0) {
    std::vector<RGB> seq_out;
    seq_out.resize(count);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    ContrastScaleSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < count; i++) {
      ASSERT_EQ(seq_out[i].R, out[i].R);
      ASSERT_EQ(seq_out[i].G, out[i].G);
      ASSERT_EQ(seq_out[i].B, out[i].B);
    }
  }
}

TEST(anikin_m_contrastscale_mpi, contrast_doesnt_change) {
  const int count = 10;
  float k = 1;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  // Seq
  if (world.rank() == 0) {
    std::vector<RGB> seq_out;
    seq_out.resize(count);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    // Create Task
    ContrastScaleSeq testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (int i = 0; i < count; i++) {
      ASSERT_EQ(in[i].R, out[i].R);
      ASSERT_EQ(in[i].G, out[i].G);
      ASSERT_EQ(in[i].B, out[i].B);
      ASSERT_EQ(in[i].R, seq_out[i].R);
      ASSERT_EQ(in[i].G, seq_out[i].G);
      ASSERT_EQ(in[i].B, seq_out[i].B);
    }
  }
}

TEST(anikin_m_contrastscale_mpi, wrong_out_count) {
  const int count = 10;
  float k = 0.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count + 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(anikin_m_contrastscale_mpi, zero_image) {
  const int count = 0;
  float k = 1.5;
  boost::mpi::communicator world;
  std::vector<RGB> in;
  std::vector<RGB> out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in.reserve(count);
    for (int i = 0; i < count; i++) {
      in.push_back(getrandomRGB());
    }
    out.resize(count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  ContrastScaleMpi testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}