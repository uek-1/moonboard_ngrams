import sys
import nltk
import matplotlib.pyplot as plt
# nltk.download('punkt')
MAX_GRAM = 2
FONT_TO_HUECO = {
    "4":0,
    "5":1,
    "5+":2,
    "6A":3,
    "6A+":3,
    "6B":4,
    "6B+":4,
    "6C": 5,
    "6C+": 5,
    "7A": 6,
    "7A+": 7,
    "7B": 8,
    "7B+": 8,
    "7C": 9,
    "7C+": 10,
    "8A": 11,
    "8A+": 12,
    "8B": 13,
    "8B+": 14,
    "8C": 15,
    "8C+": 16,
    "9A": 17,
}

from nltk.lm.api import LanguageModel
class SmoothedStupidBackoff(LanguageModel):
    def __init__(self, alpha=0.4, discount = 0.75, *args, **kwargs):
        from nltk.lm.smoothing import AbsoluteDiscounting, KneserNey

        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.smoothing = AbsoluteDiscounting(self.vocab, self.counts, discount = discount)

    def unmasked_score(self, word, context=None):
        if not context:
            # Base recursion
            s = self.smoothing.unigram_score(word)
            return min(1, s + 0.02) if s > 0 else 0.01

        counts = self.context_counts(context)
        word_count = counts[word]
        # norm_count = counts.N()
        if word_count > 0:
            alpha, gamma = self.smoothing.alpha_gamma(word, context)
            return alpha / gamma
        else:
            return self.alpha * self.unmasked_score(word, context[1:])

   

def get_routes_from_file(file_name):
    import json

    routes = {grade:[] for grade in FONT_TO_HUECO.values()}
    with open(file_name) as file:
        data = json.load(file)
        for item in data:
            hueco_grade = FONT_TO_HUECO[item['Grade']]
            routes[hueco_grade].append(item['Moves'])

    routes = {grade:probs for grade, probs in routes.items() if len(probs) > 100}

    print(list(f"{grade}:{len(probs)}" for grade,probs in routes.items()))
    
    return routes

def preprocess_data(data):
    from nltk.lm.preprocessing import padded_everygram_pipeline
    return padded_everygram_pipeline(MAX_GRAM, data)

def train_test_split(data):
    count = len(data)
    return data[:(count // 10) * 9], data[(count // 10) * 9 :]


def get_datasets(file_name):
    import copy
    train_sets = {}
    test_sets = {}
    vocabs = {}

    routes = get_routes_from_file(file_name)

    for grade in routes.keys():
        train_routes, test_routes= train_test_split(routes[grade])

        train_examples, vocabs[grade] = preprocess_data(train_routes)
        test_examples, _ = preprocess_data(test_routes)

        train_sets[grade] = [[ngram for ngram in sent] for sent in train_examples]
        test_sets[grade] = [[ngram for ngram in sent if len(ngram) == MAX_GRAM] for sent in test_examples]

        train_vocab = list(copy.deepcopy(vocabs[grade]))
        for i in range(len(test_sets[grade])):
            sent = test_sets[grade][i]

            cond = lambda word: word in train_vocab and word != "<s>" and word != "</s>"

            filtered_sent = [tuple(trigram[i] if cond(trigram[i]) else "<UNK>" for i in range(MAX_GRAM)) for trigram in sent]

            test_sets[grade][i] = filtered_sent

    return train_sets, test_sets, vocabs

def get_models(train_sets, vocabs):
    models = {}
    for name in train_sets.keys():
        model = SmoothedStupidBackoff(order = MAX_GRAM, alpha = 0.3, discount = 0.4)
        model.fit(train_sets[name], vocabs[name])
        models[name] = model
    return models

def visualize_route(route, grade):
    img = plt.imread("mb2017.png")
    route = [i for i in route if i != "<s>" and i != "</s>"]
    points = [(ord(p[0]) - 65 + 1, int(p[1:])) for p in route]
    print(points)
    plt.title(grade)
    plt.yticks(range(18), range(18))
    plt.xticks(range(12), [" "] + [chr(65 + x) for x in range(11)])
    plt.scatter([p[0] for p in points], [p[1] for p in points], color = "blue", marker = "x")
    plt.imshow(img, extent = [0, 11, 0, 18])
    plt.show()

def generate_text(prompt, models, length, seed):
    prompt = prompt.split(" ")
    print(f"\nGenerating sample text with prompt <{prompt}> and seed = {seed}")
    for name in models.keys():
        generated_text = models[name].generate(length, prompt, random_seed=seed)
        print(generated_text)
        perplexity = models[name].perplexity(list(nltk.ngrams(prompt + generated_text, MAX_GRAM)))

        total = " ".join(prompt) + " " + " ".join(generated_text)
        print(f"{name}: {total} ({perplexity})")
        visualize_route(prompt + generated_text, f"V{name}")
    

def main():
    if len(sys.argv) < 2:
        print("Enter a dataset filename (with extension)")
        return 
    
    datafile_name = sys.argv[1]

    print("Creating train and dev set...")
    train_sets, test_sets, vocabs = get_datasets(datafile_name)
    print("Training models ...")
    models = get_models(train_sets, vocabs)

   
    print("Evaluating model performance...")


    true_positives = dict([(name, 0) for name in models.keys()])
    true_negatives = dict([(name, 0) for name in models.keys()])
    false_positives = dict([(name, 0) for name in models.keys()])
    false_negatives = dict([(name, 0) for name in models.keys()])

    for name in models.keys():
        for sent in test_sets[name]:            
            predictions = [(name, model.perplexity(sent)) for name, model in models.items()]

            rankings =[name for name, _ in sorted(predictions, key = lambda x: x[1])]

            if rankings[0] == name:
                true_positives[name] += 1
                true_negatives[name] += len(rankings) - 1
            else:
                false_negatives[name] += 1
                false_positives[name] += 1
                true_negatives[name] += len(rankings) - 2 if len(rankings) >= 2 else 0

    print("Accuracy on dev set")
    for name in models.keys():
        acc = (true_positives[name] + true_negatives[name]) / (true_positives[name] + true_negatives[name] + false_negatives[name] + false_positives[name])
        print(name, f"{acc * 100}%")

    print("Perplexity on dev set")
    for name in models.keys():
        flat_test_set = [trigram for sent in test_sets[name] for trigram in sent]
        print(name, models[name].perplexity(flat_test_set))


    # TESTING ON THE TEST FILE (-t)
    
    # if testfile != None:
    #     test_sentences = get_sentences_from_file(testfile)
    #     for sentence in test_sentences: 
    #         sentence_str = " ".join(sentence)
    #         ngrams = list(nltk.ngrams(sentence, MAX_GRAM))
    #         predictions = [(name, model.perplexity(ngrams)) for name, model in models.items()]
    #         author_pred, perplexity = min(predictions, key = lambda x: x[1])
    #         print(f"<{sentence_str}>\n was written by {author_pred} with perplexity {perplexity}")

    # GENERATING TEXT (-g)

    if "-g" in sys.argv:
        prompt = "A5"
        seed = 10
        length = 5 - len(prompt)

        generate_text(prompt, models, length, seed)

    return
    

if __name__ == '__main__':
    main()
