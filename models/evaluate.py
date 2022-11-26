from utils.metric import Metric
import torch

def detokenize(pred_tensor, vocab_dict):
    idx_to_token = {v: k for k, v in vocab_dict.items()}
    # convert tensor to list
    pred_list = pred_tensor.tolist()
    sentences = []
    for sequence in pred_list:
        # remove <PAD> token
        sequence = [token for token in sequence if token not in {0, 1, 2}]
        # convert to string
        sequence = [idx_to_token[token] for token in sequence]
        sentences.append(" ".join(sequence))
    return sentences

def evaluate(model, test_dataloader, vocab_dict, device):

    pred_sentences = []
    ground_truth_sentences = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (image, formula) in enumerate(test_dataloader):
            image = image.to(device)
            formula = formula.to(device)

            pred = model.predict(image)
            pred_sentences += detokenize(pred, vocab_dict)
            ground_truth_sentences += detokenize(formula, vocab_dict)

            if batch_idx % 10 == 0:
                print('Prediction Process [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(image), len(test_dataloader.dataset),
                    100. * batch_idx / len(test_dataloader)))

    print('Prediction Finished')

    metric = Metric(pred_sentences, ground_truth_sentences)
    print(f"Bleu (unigram) score on {len(pred_sentences)} samples: {metric.corpus_bleu_score((1, 0, 0, 0))}")
    print(f"Bleu (bigram) score on {len(pred_sentences)} samples: {metric.corpus_bleu_score((0, 1, 0, 0))}")
    print(f"Bleu (trigram) score on {len(pred_sentences)} samples: {metric.corpus_bleu_score((0, 0, 1, 0))}")
    print(f"Bleu (4-gram) score on {len(pred_sentences)} samples: {metric.corpus_bleu_score((0, 0, 0, 1))}")
    print(f"Bleu (1-4-gram) score on {len(pred_sentences)} samples: {metric.corpus_bleu_score((0.25, 0.25, 0.25, 0.25))}")
    print(f"Edit distance score on {len(pred_sentences)} samples: {metric.result(metric='edit_distance')}")
