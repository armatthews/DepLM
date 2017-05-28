#include <iostream>
#include <csignal>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>
#include "deplm.h"
#include "utils.h"
#include "io.h"

using namespace dynet;
using namespace dynet::expr;
using namespace std;
namespace po = boost::program_options;

KBestList<shared_ptr<OutputSentence>> DoBeamSearch(OutputModel* output_model, unsigned K, unsigned beam_size, unsigned max_length, float length_bonus) {
  assert (beam_size >= K);
  ComputationGraph cg;
  output_model->NewGraph(cg);

  KBestList<shared_ptr<OutputSentence>> complete_hyps(K);
  KBestList<pair<shared_ptr<OutputSentence>, RNNPointer>> top_hyps(beam_size);
  top_hyps.add(0.0, make_pair(make_shared<OutputSentence>(), output_model->GetStatePointer()));

  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<pair<shared_ptr<OutputSentence>, RNNPointer>> new_hyps(beam_size);

    for (auto& hyp : top_hyps.hypothesis_list()) {
      double hyp_score = get<0>(hyp);

      // Early termination: if we have K completed hypotheses, and the current prefix's
      // score is worse than the worst of the complete hyps, then it's impossible to
      // later get a better hypothesis later.
      // Assumption: the log prob of every word will be <= buffer.
      // This is true for 0.0 by default, but length priors and such may invalidate this assumption.
      const float buffer = length_bonus;
      if (complete_hyps.size() >= K && hyp_score < complete_hyps.worst_score() - buffer) {
        break;
      }

      if (new_hyps.size() >= K && hyp_score < new_hyps.worst_score() - buffer) {
        break;
      }

      shared_ptr<OutputSentence> hyp_sentence = get<0>(get<1>(hyp));
      RNNPointer state_pointer = get<1>(get<1>(hyp));
      assert (hyp_sentence->size() == length);
      Expression output_state = output_model->GetState(state_pointer);
      KBestList<shared_ptr<Word>> best_words = output_model->PredictKBest(state_pointer, beam_size);

      for (auto& w : best_words.hypothesis_list()) {
        double word_score = get<0>(w);
        shared_ptr<Word> word = get<1>(w);
        double new_score = hyp_score + word_score;
        shared_ptr<OutputSentence> new_sentence(new OutputSentence(*hyp_sentence));
        new_sentence->push_back(word);
        output_model->AddInput(word, state_pointer);
        if (!output_model->IsDone()) {
          new_score += length_bonus;
          new_hyps.add(new_score, make_pair(new_sentence, output_model->GetStatePointer()));
        }
        else {
          complete_hyps.add(new_score, new_sentence);
        }
      }
    }
    top_hyps = new_hyps;
  }

  for (auto& hyp : top_hyps.hypothesis_list()) {
    double score = get<0>(hyp);
    shared_ptr<OutputSentence> sentence = get<0>(get<1>(hyp));
    complete_hyps.add(score, sentence);
  }
  return complete_hyps;
}

void OutputKBestList(unsigned sentence_number, KBestList<shared_ptr<OutputSentence>> kbest, Dict& vocab) {
  for (auto& scored_hyp : kbest.hypothesis_list()) {
    double score = scored_hyp.first;
    const shared_ptr<OutputSentence> hyp = scored_hyp.second;
    vector<string> words(hyp->size());
    for (unsigned i = 0; i < hyp->size(); ++i) {
      words[i] = vocab.convert(dynamic_pointer_cast<StandardWord>(hyp->at(i))->id);
    }
    string translation = boost::algorithm::join(words, " ");
    cout << sentence_number << " ||| " << translation << " ||| " << score << endl;
  }
  cout.flush();
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("model", po::value<string>(), "Trained model")
  ("kbest_size,k", po::value<unsigned>()->default_value(1), "K-best list size")
  ("beam_size,b", po::value<unsigned>()->default_value(10), "Beam size")
  ("max_length", po::value<unsigned>()->default_value(100), "Maximum length of output sentences")
  ("length_bonus", po::value<float>()->default_value(0.0f), "Length bonus per word");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
  positional_options.add("text", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  Dict vocab;
  Model dynet_model;
  DependencyOutputModel* model = new DependencyOutputModel();
  Trainer* trainer = nullptr;

  const string model_filename = vm["model"].as<string>();
  const unsigned kbest_size = vm["kbest_size"].as<unsigned>();
  const unsigned beam_size = vm["beam_size"].as<unsigned>();
  const unsigned max_length = vm["max_length"].as<unsigned>();
  const float length_bonus = vm["length_bonus"].as<float>();
  Deserialize(model_filename, vocab, *model, dynet_model, trainer);

  KBestList<shared_ptr<OutputSentence>> kbest = DoBeamSearch(model, kbest_size, beam_size, max_length, length_bonus);
  OutputKBestList(0, kbest, vocab);

  return 0;
}
