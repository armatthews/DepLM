#pragma once
#include <boost/serialization/access.hpp>
#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "embedder.h"
#include "kbestlist.h"
#include "utils.h"
#include "mlp.h"

class OutputModel {
public:
  virtual ~OutputModel();

  virtual void NewGraph(ComputationGraph& cg) = 0;
  virtual void SetDropout(float rate) {}
  virtual Expression GetState() const;
  virtual Expression GetState(RNNPointer p) const = 0;
  virtual RNNPointer GetStatePointer() const = 0;
  virtual Expression AddInput(const shared_ptr<const Word> prev_word);
  virtual Expression AddInput(const shared_ptr<const Word> prev_word, const RNNPointer& p) = 0;

  virtual Expression PredictLogDistribution();
  virtual Expression PredictLogDistribution(RNNPointer p) = 0;
  virtual KBestList<shared_ptr<Word>> PredictKBest(unsigned K);
  virtual KBestList<shared_ptr<Word>> PredictKBest(RNNPointer p, unsigned K) = 0;
  virtual pair<shared_ptr<Word>, float> Sample();
  virtual pair<shared_ptr<Word>, float> Sample(RNNPointer p) = 0;
  virtual Expression Loss(const shared_ptr<const Word> ref);
  virtual Expression Loss(RNNPointer p, const shared_ptr<const Word> ref) = 0;

  virtual bool IsDone() const;
  virtual bool IsDone(RNNPointer p) const = 0;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {}
};

class DependencyOutputModel : public OutputModel {
public:
  DependencyOutputModel();
  DependencyOutputModel(Model& model, Embedder* embedder, unsigned state_dim, unsigned final_hidden_dim, Dict& vocab);

  Expression BuildGraph(const OutputSentence& sent);

  void NewGraph(ComputationGraph& cg) override;
  void SetDropout(float rate) override;
  Expression GetState(RNNPointer p) const override;
  RNNPointer GetStatePointer() const override;
  Expression AddInput(const shared_ptr<const Word> prev_word, const RNNPointer& p) override;

  Expression PredictLogDistribution(RNNPointer p) override;
  KBestList<shared_ptr<Word>> PredictKBest(RNNPointer p, unsigned K) override;
  pair<shared_ptr<Word>, float> Sample(RNNPointer p) override;
  Expression Loss(RNNPointer p, const shared_ptr<const Word> ref) override;
  bool IsDone(RNNPointer p) const override;

  const Dict* vocab; // XXX: Remove me
  vector<string> stack_strings; // XXX: Remove me
  vector<string> comp_strings; // XXX: Remove me

private:
  typedef tuple<RNNPointer, RNNPointer, unsigned, bool> State; // Stack pointer, comp pointer, stack depth, done with left

  Embedder* embedder;
  GRUBuilder stack_lstm;
  GRUBuilder comp_lstm;
  MLP final_mlp;

  Parameter emb_transform_p; // Simple linear transform from word embedding space to state space
  Parameter stack_lstm_init_p;
  Parameter comp_lstm_init_p;

  Expression emb_transform;
  vector<Expression> stack_lstm_init;
  vector<Expression> comp_lstm_init;

  unsigned half_state_dim;
  unsigned done_with_left;
  unsigned done_with_right;

  vector<State> prev_states;
  vector<RNNPointer> stack; // From each state, if you were to see </RIGHT> where would you go back to?
  vector<RNNPointer> head;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & boost::serialization::base_object<OutputModel>(*this);
    ar & embedder;
    ar & stack_lstm;
    ar & comp_lstm;
    ar & final_mlp;

    ar & emb_transform_p;
    ar & stack_lstm_init_p;
    ar & comp_lstm_init_p;

    ar & half_state_dim;
    ar & done_with_left;
    ar & done_with_right;
  }
};
BOOST_CLASS_EXPORT_KEY(DependencyOutputModel)
