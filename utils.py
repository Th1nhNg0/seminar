from tqdm import tqdm
from underthesea import sent_tokenize

def segment_documents(docs, max_length=6, stride=3):
    """
    Segment documents into smaller chunks
    """
    segments = []
    for doc in tqdm(docs):
        sentences = sent_tokenize(doc)
        for i in range(0, len(sentences), stride):
            segment = ' '.join(sentences[i:i+max_length])
            segments.append(segment)
    return segments



def compute_exact_match(prediction, truth):
    return int(prediction == truth)

def compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)