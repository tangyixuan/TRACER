import json


def precision_recall_f1(y_true, y_pred, label):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true, pred in zip(y_true, y_pred):
        if true == label and pred == label:
            true_positives += 1
        elif pred == label and true != label:
            false_positives += 1
        elif true == label and pred != label:
            false_negatives += 1

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def macro_f1(y_true, y_pred, labels):
    f1_scores = []
    for label in labels:
        _, _, f1 = precision_recall_f1(y_true, y_pred, label)
        f1_scores.append(f1)
    return sum(f1_scores) / len(labels)


def main():
    datapath = "method/result/20250413_000157/log.jsonl"
    labels_set = ["false", "half-true", "true"]

    y_true = []
    y_pred = []

    with open(datapath, 'r') as fread:
        for line in fread:
            data = json.loads(line)
            y_pred.append(data['pred'])
            y_true.append(data['veracity'])

    acc = accuracy(y_true, y_pred)
    macro_f1_score = macro_f1(y_true, y_pred, labels_set)
    half_true_precision, half_true_recall, half_true_f1 = precision_recall_f1(y_true, y_pred, "half-true")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1_score:.4f}")
    print(f"Half-true Precision: {half_true_precision:.4f}")
    print(f"Half-true Recall: {half_true_recall:.4f}")
    print(f"Half-true F1: {half_true_f1:.4f}")


if __name__ == "__main__":
    main()
    