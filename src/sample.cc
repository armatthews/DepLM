#include <iostream>
#include <csignal>
#include <boost/algorithm/string/join.hpp>
#include "train.h"
#include "deplm.h"
#include "utils.h"
#include "io.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

tuple<OutputSentence, float> Sample(OutputModel* model, unsigned max_length, Dict& vocab) {
  OutputSentence sent;
  float total_loss = 0.0f;
  for (unsigned i = 0; i < max_length; ++i) {
    shared_ptr<Word> word;
    float word_loss;
    tie(word, word_loss) = model->Sample();
    sent.push_back(word);
    total_loss += word_loss;

    model->AddInput(word);
    if (model->IsDone()) {
      break;
    }
  }

  return make_pair(sent, total_loss);
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("model", po::value<string>()->required(), "Trained model whose grammar will be dumped")
  ("max_length", po::value<unsigned>()->default_value(300), "Maximum length of output sentences");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("model", 1);

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
  const unsigned max_length = vm["max_length"].as<unsigned>();
  Deserialize(model_filename, vocab, *model, dynet_model, trainer);

  while(true) {
    ComputationGraph cg;
    model->NewGraph(cg);
    OutputSentence sample;
    float loss;
    tie(sample, loss) = Sample(model, max_length, vocab);

    vector<string> words(sample.size());
    for (unsigned i = 0; i < sample.size(); ++i) {
      words[i] = vocab.convert(dynamic_pointer_cast<StandardWord>(sample.at(i))->id);
    }
    string sample_string = boost::algorithm::join(words, " ");

    cout << loss << " ||| " << sample_string << endl;
  }

  return 0;
}
