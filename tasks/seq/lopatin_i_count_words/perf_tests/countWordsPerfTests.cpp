#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>

#include "core/perf/include/perf.hpp"
#include "seq/lopatin_i_count_words/include/countWordsSeqHeader.hpp"

std::string testData =
    "EVABOY HAS BEEN FOUND ALIVE IN BAGUIO. FOLLOWING THE UNEXPECTED POPULARITY OF HIS NXC WORKS 'CHOW GARDENS'"
    " AND MID-WAY THROUGH WORKING ON THE UNFINISHED ALBUM 'JOOJ TIL I DIE',"
    "FILIPINO-AMERICAN V-MUSICIAN EVABOY HAD A TOTAL MENTAL BREAKDOWN AND WENT INTO ISOLATION. WITH THE HELP OF "
    "VIRTUAL DJ 'GANJINCMANE', PREVIOUSLY KNOWN AS 'DJ GAPE',"
    "EVABOY FAKED HIS DEATH, AS THE JEALOUS INTERNET SOUNDCLOUD COLLECTIVES FORMED BY 15 YEAR OLD EVANGELION FANS ON "
    "DISCORD PUT A BOUNTY ON EVABOY'S HEAD, SIFTING THROUGH ANY INFO THEY CAN TO FIND A REASON TO CALL OUR BELOVED "
    "EVABOY OUT AND DEEM HIM PROBLEMATIC. SINCE THEN, EVABOY HAS BEEN HIDING AWAY IN BAGUIO PHILIPPINES, HOMELESS, "
    "LIVING IN HIS CAR, AND WORKING ON A NEW ALBUM WITH A MICROPHONE, SYNTH, AND SAMPLER HE BUILT FROM SCRATCH USING "
    "METAL SCRAPS HE WOULD FIND IN THA TRASH. UNFORTUNATELY (OR FORTUNATELY) FOR GANJINCMANE, HE HAD A BRIEF STAY IN "
    "JAIL UNTIL IT WAS DISCOVERED THAT THE KILLING WAS AN ORCHESTRATED RUSE. GANJINCMANE IS CURRENTLY ON PROBATION, "
    "HOSTING A RADIO SHOW, AND EVABOY AND HIM ARE STILL BEST BUDS.THIS ALBUM IS PARTLY THE COMPLETED MISSING PIECES OF "
    "THE PREVIOUS EVABOY ALBUM, AND ADDITIONALLY THE RESULT OF ~2 YEARS OF LAYING LOW, BEING VINDICTIVE AND BITTER, "
    "SOMETIMES SELF REFLECTING TO A POINT OF TOTAL SELF HATRED, AND INCORRECTLY DEALING WITH A LIFE RUINING AND "
    "DEBILITATING MENTAL ILLNESS THAT ONLY WORKED TO FURTHER BURN EVABOY AND THOSE AROUND HIM. AND AS SO, THIS NEW "
    "ALBUM IS TITLED 'BEEF'."
    "BECAUSE THIS ALBUM IS ABOUT THE BEEF. THIS ALBUM IS BEEF. ANY BEEF. MY BEEF. YOUR BEEF. FOOD BEEF. THIS IS "
    "BEEF. IF YOU HAVE BEEF, THIS IS THE ALBUM YOU BEEF TO. GET READY TO BEEF. EVABOY HAS RETURNED. FUCK YOU AND DIE."
    "EVABOY HAS BEEN FOUND ALIVE IN BAGUIO. FOLLOWING THE UNEXPECTED POPULARITY OF HIS NXC WORKS 'CHOW GARDENS'"
    " AND MID-WAY THROUGH WORKING ON THE UNFINISHED ALBUM 'JOOJ TIL I DIE',"
    "FILIPINO-AMERICAN V-MUSICIAN EVABOY HAD A TOTAL MENTAL BREAKDOWN AND WENT INTO ISOLATION. WITH THE HELP OF "
    "VIRTUAL DJ 'GANJINCMANE', PREVIOUSLY KNOWN AS 'DJ GAPE',"
    "EVABOY FAKED HIS DEATH, AS THE JEALOUS INTERNET SOUNDCLOUD COLLECTIVES FORMED BY 15 YEAR OLD EVANGELION FANS ON "
    "DISCORD PUT A BOUNTY ON EVABOY'S HEAD, SIFTING THROUGH ANY INFO THEY CAN TO FIND A REASON TO CALL OUR BELOVED "
    "EVABOY OUT AND DEEM HIM PROBLEMATIC. SINCE THEN, EVABOY HAS BEEN HIDING AWAY IN BAGUIO PHILIPPINES, HOMELESS, "
    "LIVING IN HIS CAR, AND WORKING ON A NEW ALBUM WITH A MICROPHONE, SYNTH, AND SAMPLER HE BUILT FROM SCRATCH USING "
    "METAL SCRAPS HE WOULD FIND IN THA TRASH. UNFORTUNATELY (OR FORTUNATELY) FOR GANJINCMANE, HE HAD A BRIEF STAY IN "
    "JAIL UNTIL IT WAS DISCOVERED THAT THE KILLING WAS AN ORCHESTRATED RUSE. GANJINCMANE IS CURRENTLY ON PROBATION, "
    "HOSTING A RADIO SHOW, AND EVABOY AND HIM ARE STILL BEST BUDS.THIS ALBUM IS PARTLY THE COMPLETED MISSING PIECES OF "
    "THE PREVIOUS EVABOY ALBUM, AND ADDITIONALLY THE RESULT OF ~2 YEARS OF LAYING LOW, BEING VINDICTIVE AND BITTER, "
    "SOMETIMES SELF REFLECTING TO A POINT OF TOTAL SELF HATRED, AND INCORRECTLY DEALING WITH A LIFE RUINING AND "
    "DEBILITATING MENTAL ILLNESS THAT ONLY WORKED TO FURTHER BURN EVABOY AND THOSE AROUND HIM. AND AS SO, THIS NEW "
    "ALBUM IS TITLED 'BEEF'."
    "BECAUSE THIS ALBUM IS ABOUT THE BEEF. THIS ALBUM IS BEEF. ANY BEEF. MY BEEF. YOUR BEEF. FOOD BEEF. THIS IS "
    "BEEF. IF YOU HAVE BEEF, THIS IS THE ALBUM YOU BEEF TO. GET READY TO BEEF. EVABOY HAS RETURNED. FUCK YOU AND DIE.";

// text for testing with 571 words

TEST(word_count_seq, test_pipeline_run) {
  std::string input = testData;
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
  taskData->outputs_count.emplace_back(word_count.size());

  auto testTask = std::make_shared<lopatin_i_count_words_seq::TestTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(571, word_count[0]);
}

TEST(word_count_seq, test_task_run) {
  std::string input = testData;
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
  taskData->outputs_count.emplace_back(word_count.size());

  auto testTask = std::make_shared<lopatin_i_count_words_seq::TestTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(571, word_count[0]);
}