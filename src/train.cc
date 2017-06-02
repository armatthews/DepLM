#include <iostream>
#include <csignal>
#include <chrono>
#include "train.h"
#include "deplm.h"
#include "utils.h"
#include "io.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

class Learner : public ILearner<OutputSentence, SufficientStats> {
public:
  Learner(Dict& vocab, DependencyOutputModel& model, Model& dynet_model, Trainer* trainer) : vocab(vocab), model(model), dynet_model(dynet_model), trainer(trainer) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const OutputSentence& datum, bool learn) {
    ComputationGraph cg;
    model.NewGraph(cg);

    if (learn) {
      model.SetDropout(dropout_rate);
    }
    else {
      model.SetDropout(0.0f);
    }

    Expression loss_expr = model.BuildGraph(datum);
    dynet::real loss = as_scalar(cg.forward(loss_expr));
    if (learn) {
      cg.backward(loss_expr);
    }
    return SufficientStats(loss, datum.size(), 1);
  }

  void SaveModel() {
    if (!quiet) {
      Serialize(vocab, model, dynet_model, trainer);
    }
  }

  bool quiet;
  float dropout_rate;
private:
  Dict& vocab;
  DependencyOutputModel& model;
  Model& dynet_model;
  Trainer* trainer;
};

// This function lets us elegantly handle the user pressing ctrl-c.
// We set a global flag, which causes the training loops to clean up
// and break. In particular, this allows models to be saved to disk
// before actually exiting the program.
bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
    dynet::mp::stop_requested = true;
  }
}

SufficientStats RunDevSet(vector<OutputSentence>& dev_set, Learner* learner) {
  SufficientStats r;
  for (unsigned i = 0; i < dev_set.size(); ++i) {
    r += learner->LearnFromDatum(dev_set[i], false);
  }
  return r;
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  cerr << "Invoked as:";
  for (int i = 0; i < argc; ++i) {
    cerr << " " << argv[i];
  }
  cerr << "\n";

  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("train_text", po::value<string>()->required(), "Training text")
  ("dev_text", po::value<string>()->required(), "Dev text, used for early stopping")
  ("hidden_dim,h", po::value<unsigned>()->default_value(64), "Size of hidden layers")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("dropout_rate", po::value<float>()->default_value(0.0), "Dropout rate (should be >= 0.0 and < 1)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best")
  ("quiet,q", "Do not output model")
  ("model", po::value<string>(), "Reload this model and continue learning");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("train_text", 1);
  positional_options.add("dev_text", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);
 
  const unsigned num_cores = vm["cores"].as<unsigned>();  
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const string train_text_filename = vm["train_text"].as<string>();
  const string dev_text_filename = vm["dev_text"].as<string>();

  Dict vocab;
  Model dynet_model;
  DependencyOutputModel* model = nullptr;
  Trainer* trainer = nullptr;

  if (vm.count("model")) {
    string model_filename = vm["model"].as<string>();
    model = new DependencyOutputModel();
    Deserialize(model_filename, vocab, *model, dynet_model, trainer);
    assert (vocab.is_frozen());

    if (vm.count("sgd") || vm.count("adagrad") || vm.count("adam") || vm.count("rmsprop") || vm.count("momentum")) {
      trainer = CreateTrainer(dynet_model, vm);
    }
  }

  vector<OutputSentence> train_text = ReadText(train_text_filename, vocab);

  if (!vm.count("model")) {
    unsigned hidden_dim = vm["hidden_dim"].as<unsigned>();
    Embedder* embedder = new StandardEmbedder(dynet_model, vocab.size(), hidden_dim);
    model = new DependencyOutputModel(dynet_model, embedder, hidden_dim, hidden_dim, vocab);
    vocab.freeze();
    vocab.set_unk("UNK");
  }

  vector<OutputSentence> dev_text = ReadText(dev_text_filename, vocab);

  cerr << "Vocabulary size: " << vocab.size() << endl;
  cerr << "Total parameters: " << dynet_model.parameter_count() << endl;

  trainer = CreateTrainer(dynet_model, vm);
  Learner learner(vocab, *model, dynet_model, trainer);
  learner.quiet = vm.count("quiet") > 0;
  learner.dropout_rate = vm["dropout_rate"].as<float>();

  const bool quiet = vm.count("quiet") > 0;
  const unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  const unsigned report_frequency = vm["report_frequency"].as<unsigned>(); 

  unsigned data_since_dev = 0;
  unsigned data_since_report = 0;
  SufficientStats loss;
  SufficientStats best_dev_stats;
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

  for (unsigned iteration = 0; iteration < num_iterations; ++iteration) {
    for (unsigned i = 0; i < train_text.size(); ++i) {
      const OutputSentence& sentence = train_text[i];
      loss += learner.LearnFromDatum(sentence, true);

      float fractional_epoch = iteration + 1.0f * (i + 1) / train_text.size();
      if (++data_since_report == report_frequency) {
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        double secs = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;
        start_time = end_time;
        cerr << fractional_epoch << "\t" << "loss = " << loss << " (" << secs << " secs)" << endl;
        data_since_report = 0;
        loss = SufficientStats();
      }
      trainer->update();

      if (++data_since_dev == dev_frequency) {
        SufficientStats dev_stats = RunDevSet(dev_text, &learner);
        bool new_best = (best_dev_stats.sentence_count == 0) || (dev_stats < best_dev_stats);
        cerr << fractional_epoch << "\t" << "dev loss = " << dev_stats << (new_best ? " (New best!)" : "") << endl;
        if (new_best) {
          if (!quiet) {
            Serialize(vocab, *model, dynet_model, trainer);
          }
          best_dev_stats = dev_stats;
        }
        data_since_dev = 0;
      }
    }
  }

  return 0;
}
