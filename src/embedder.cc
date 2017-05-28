#include "embedder.h"
BOOST_CLASS_EXPORT_IMPLEMENT(StandardEmbedder)

const unsigned lstm_layer_count = 2;

void Embedder::NewGraph(ComputationGraph& cg) {}
void Embedder::SetDropout(float) {}

StandardEmbedder::StandardEmbedder() {}

StandardEmbedder::StandardEmbedder(Model& model, unsigned vocab_size, unsigned emb_dim) : emb_dim(emb_dim), pcg(nullptr) {
  embeddings = model.add_lookup_parameters(vocab_size, {emb_dim});
}

void StandardEmbedder::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
}

void StandardEmbedder::SetDropout(float) {}

unsigned StandardEmbedder::Dim() const {
  return emb_dim;
}

Expression StandardEmbedder::Embed(const shared_ptr<const Word> word) {
  const shared_ptr<const StandardWord> standard_word = dynamic_pointer_cast<const StandardWord>(word);
  assert (standard_word != nullptr);
  return lookup(*pcg, embeddings, standard_word->id);
}
