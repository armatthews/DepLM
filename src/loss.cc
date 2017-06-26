#include <iostream>
#include <csignal>
#include "train.h"
#include "deplm.h"
#include "utils.h"
#include "io.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("model", po::value<string>()->required(), "Trained model whose grammar will be dumped")
  ("verbose", "Verbose word-level output")
  ("text", po::value<string>()->required(), "Input text");

  AddTrainerOptions(desc);

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

  const bool verbose = vm.count("verbose") > 0;
  const string model_filename = vm["model"].as<string>();
  const string text_filename = vm["text"].as<string>();
  Deserialize(model_filename, vocab, *model, dynet_model, trainer);
  model->vocab = &vocab;
  vector<OutputSentence> input_text = ReadText(text_filename, vocab);

  for (unsigned i = 0; i < input_text.size(); ++i) {
    ComputationGraph cg;
    model->NewGraph(cg);
    if (verbose) {
      float total_loss = 0.0f;
      cout << fixed;
      cout.precision(4);
      for (unsigned j = 0; j < input_text[i].size(); ++j) {
        const shared_ptr<const Word> word = input_text[i][j];
        const string word_str = vocab.convert(dynamic_pointer_cast<const StandardWord>(word)->id);
        RNNPointer p = model->GetStatePointer();
        float loss = as_scalar(model->Loss(p, word).value());
        KBestList<shared_ptr<Word>> alternatives = model->PredictKBest(p, 3);
        cout << i << "\t" << j << "\t" << word_str << (word_str.length() < 8 ? "\t" : "") << "\t" << loss << "\t";
        for (auto& kv : alternatives.hypothesis_list()) {
          double score = get<0>(kv);
          auto alternative = dynamic_pointer_cast<const StandardWord>(get<1>(kv));
          cout << vocab.convert(alternative->id) << " (" << score << ") ";
        }
        cout << endl;

        model->AddInput(word, p);
      }
      cout << endl;
    }
    else {
      Expression loss_expr = model->BuildGraph(input_text[i]);
      float loss = as_scalar(loss_expr.value());
      cout << i << " ||| " << loss << endl;
    }
  }

  return 0;
}
