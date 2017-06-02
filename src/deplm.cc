#include <boost/algorithm/string/predicate.hpp>
#include "deplm.h"

const unsigned lstm_layer_count = 2;

OutputModel::~OutputModel() {}

bool OutputModel::IsDone() const {
  return IsDone(GetStatePointer());
}

Expression OutputModel::GetState() const {
  return GetState(GetStatePointer());
}

Expression OutputModel::AddInput(const shared_ptr<const Word> prev_word) {
  return AddInput(prev_word, GetStatePointer());
}

Expression OutputModel::PredictLogDistribution() {
  return PredictLogDistribution(GetStatePointer());
}

KBestList<shared_ptr<Word>> OutputModel::PredictKBest(unsigned K) {
  return PredictKBest(GetStatePointer(), K);
}

pair<shared_ptr<Word>, float> OutputModel::Sample() {
  return Sample(GetStatePointer());
}

Expression OutputModel::Loss(const shared_ptr<const Word> ref) {
  return Loss(GetStatePointer(), ref);
}

DependencyOutputModel::DependencyOutputModel() {}

DependencyOutputModel::DependencyOutputModel(Model& model, Embedder* embedder, unsigned state_dim, unsigned final_hidden_dim, Dict& vocab) {
  assert (state_dim % 2 == 0);
  const unsigned vocab_size = vocab.size();
  half_state_dim = state_dim / 2;

  this->embedder = embedder;
  stack_lstm = LSTMBuilder(lstm_layer_count, half_state_dim, half_state_dim, model);
  comp_lstm = LSTMBuilder(lstm_layer_count, half_state_dim, half_state_dim, model);
  final_mlp = MLP(model, 2 * half_state_dim, final_hidden_dim, vocab_size);

  emb_transform_p = model.add_parameters({half_state_dim, embedder->Dim()});
  stack_lstm_init_p = model.add_parameters({lstm_layer_count * 2 * half_state_dim});
  comp_lstm_init_p = model.add_parameters({lstm_layer_count * 2 * half_state_dim});

  done_with_left = vocab.convert("</LEFT>");
  done_with_right = vocab.convert("</RIGHT>");
}

void DependencyOutputModel::NewGraph(ComputationGraph& cg) {
  embedder->NewGraph(cg);
  stack_lstm.new_graph(cg);
  comp_lstm.new_graph(cg);
  final_mlp.NewGraph(cg);

  emb_transform = parameter(cg, emb_transform_p);
  stack_lstm_init = MakeLSTMInitialState(parameter(cg, stack_lstm_init_p), half_state_dim, lstm_layer_count);
  comp_lstm_init = MakeLSTMInitialState(parameter(cg, comp_lstm_init_p), half_state_dim, lstm_layer_count);

  stack_lstm.start_new_sequence(stack_lstm_init);
  comp_lstm.start_new_sequence(comp_lstm_init);

  prev_states.clear();
  prev_states.push_back(make_tuple(stack_lstm.state(), comp_lstm.state(), 0, true));

  head.clear();
  head.push_back((RNNPointer)-1);

  stack.clear();
  stack.push_back((RNNPointer)-1);
}

Expression DependencyOutputModel::BuildGraph(const OutputSentence& sent) {
  vector<Expression> losses;
  for (unsigned i = 0; i < sent.size(); ++i) {
    const shared_ptr<const Word> word = sent[i];
    Expression loss = Loss(GetStatePointer(), word);
    losses.push_back(loss);

    AddInput(sent[i], GetStatePointer());
  }
  return sum(losses);
}

void DependencyOutputModel::SetDropout(float rate) {
  stack_lstm.set_dropout(rate);
  comp_lstm.set_dropout(rate);
  final_mlp.SetDropout(rate);
}

Expression DependencyOutputModel::GetState(RNNPointer p) const {
  RNNPointer stack_pointer;
  RNNPointer comp_pointer;
  unsigned stack_depth;
  bool left_done;
  tie(stack_pointer, comp_pointer, stack_depth, left_done) = prev_states[p];
  Expression stack_state = stack_lstm.get_h(stack_pointer).back();
  Expression comp_state = comp_lstm.get_h(comp_pointer).back();
  return concatenate({stack_state, comp_state});
}

RNNPointer DependencyOutputModel::GetStatePointer() const {
  return (RNNPointer)((int)prev_states.size() - 1);
}

Expression DependencyOutputModel::AddInput(const shared_ptr<const Word> prev_word, const RNNPointer& p) {
  assert (prev_states.size() == stack.size());
  assert (prev_states.size() == head.size());
  assert (p < prev_states.size());

  unsigned wordid = dynamic_pointer_cast<const StandardWord>(prev_word)->id;
  Expression embedding = embedder->Embed(prev_word);
  Expression transformed_embedding = emb_transform * embedding;

  RNNPointer stack_pointer;
  RNNPointer comp_pointer;
  unsigned stack_depth;
  bool left_done;
  tie(stack_pointer, comp_pointer, stack_depth, left_done) = prev_states[p];
  RNNPointer parent = (RNNPointer)-1337;

  Expression input_vec = transformed_embedding;

  if (wordid == done_with_right) {
    assert (left_done);
    Expression node_repr = comp_lstm.add_input(comp_pointer, input_vec);

    RNNPointer pop_to_i = stack[p];
    if (pop_to_i == -1) {
      stack_pointer = (RNNPointer)-1;
      comp_lstm.add_input((RNNPointer)-1, node_repr);
      comp_pointer = comp_lstm.state();
      stack_depth --;
      left_done = true;
      parent = -1;
    }
    else {
      State& pop_to = prev_states[pop_to_i];

      stack_pointer = stack_lstm.get_head(stack_pointer);
      assert (stack_pointer == get<0>(pop_to));

      comp_lstm.add_input(get<1>(pop_to), node_repr);
      comp_pointer = comp_lstm.state();

      stack_depth--;
      assert (stack_depth == get<2>(pop_to));

      left_done = get<3>(pop_to);

      parent = stack[pop_to_i];
    }
  }
  else if (wordid == done_with_left) {
    assert (!left_done);
    comp_lstm.add_input(comp_pointer, input_vec);
    comp_pointer = comp_lstm.state();
    parent = stack[p];
    left_done = true;
  }
  else {
    stack_lstm.add_input(stack_pointer, input_vec);
    comp_lstm.add_input((RNNPointer)-1, input_vec);

    stack_pointer = stack_lstm.state();
    comp_pointer = comp_lstm.state();
    stack_depth++;
    left_done = false;
    parent = p;
  }

  /*cerr << prev_states.size() << "\t" << "head: " << p << ", " << "stack: " << parent;
  cerr << ", " << "sp: " << stack_pointer << ", " << "cp: " << comp_pointer;
  cerr << ", " << "sd: " << stack_depth << ", " << "ld: " << left_done << ", " << "word: " << word << endl;*/
  stack.push_back(parent);
  head.push_back(p);
  prev_states.push_back(make_tuple(stack_pointer, comp_pointer, stack_depth, left_done));

  assert (prev_states.size() == stack.size());
  assert (prev_states.size() == head.size());
  return OutputModel::GetState();
}

Expression DependencyOutputModel::PredictLogDistribution(RNNPointer p) {
  Expression state = GetState(p);
  Expression scores = final_mlp.Feed(state);
  Expression log_probs = log_softmax(scores);
  return log_probs;
}

KBestList<shared_ptr<Word>> DependencyOutputModel::PredictKBest(RNNPointer p, unsigned K) {
  vector<float> log_probs = as_vector(PredictLogDistribution(p).value());
  unsigned stack_depth = get<2>(prev_states[p]);
  bool left_done = get<3>(prev_states[p]);

  KBestList<shared_ptr<Word>> kbest(K);
  for (unsigned i = 0; i < log_probs.size(); ++i) {
    if (i == done_with_left && (left_done || stack_depth >= 100)) {
      continue;
    }
    else if (i == done_with_right && (IsDone(p) || !left_done)) {
      continue;
    }
    shared_ptr<Word> word = make_shared<StandardWord>(i);
    kbest.add(log_probs[i], word);
  }

  return kbest;
}

pair<shared_ptr<Word>, float> DependencyOutputModel::Sample(RNNPointer p) {
  Expression state = GetState(p);
  Expression probs = softmax(final_mlp.Feed(state));
  vector<float> ps = as_vector(probs.value());

  // Zero out the probability of illegal actions
  RNNPointer stack_pointer;
  RNNPointer comp_pointer;
  unsigned stack_depth;
  bool left_done;
  tie(stack_pointer, comp_pointer, stack_depth, left_done) = prev_states[p];
  if (left_done || stack_depth >= 100) {
    ps[done_with_left] = 0.0f;
  }
  if (IsDone(p) || !left_done) {
    ps[done_with_right] = 0.0f;
  }

  // Renormalize
  float s = 0.0f;
  for (float prob : ps) {
    s += prob;
  }
  for (unsigned i = 0; i < ps.size(); ++i) {
    ps[i] /= s;
  }

  unsigned id = ::Sample(ps);
  shared_ptr<Word> w = make_shared<StandardWord>(id);
  return make_pair(w, log(ps[id]));
}

Expression DependencyOutputModel::Loss(RNNPointer p, const shared_ptr<const Word> ref) {
  Expression state = GetState(p);
  Expression log_probs = final_mlp.Feed(state);
  return pickneglogsoftmax(log_probs, dynamic_pointer_cast<const StandardWord>(ref)->id);
}

bool DependencyOutputModel::IsDone(RNNPointer p) const {
  return (get<2>(prev_states[p]) == (unsigned)-1);
}
