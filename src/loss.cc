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

  const string model_filename = vm["model"].as<string>();
  const string text_filename = vm["text"].as<string>();
  Deserialize(model_filename, vocab, *model, dynet_model, trainer);

  vector<OutputSentence> input_text = ReadText(text_filename, vocab);

  for (unsigned i = 0; i < input_text.size(); ++i) {
    ComputationGraph cg;
    model->NewGraph(cg);
    Expression loss_expr = model->BuildGraph(input_text[i]);
    float loss = as_scalar(loss_expr.value());
    cout << i << " ||| " << loss << endl;
  }

  return 0;
}
