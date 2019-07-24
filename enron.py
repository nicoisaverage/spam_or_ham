import os

# import our classifier, assumed to be in same directory
from classifier import NBC


def train(corpus='corpus'):
    classifier = NBC(filename='enron.kct')
    curdir = os.path.dirname(__file__)

    # paths to spam and ham documents
    spam_dir = os.path.join(curdir, corpus, 'spam')
    ham_dir = os.path.join(curdir, corpus, 'ham')

    # train the classifier with the spam documents
    train_category(classifier, spam_dir, 'spam')

    # train the classifier with the ham documents
    train_category(classifier, ham_dir, 'ham')

    return classifier


def train_category(classifier, path, label):
    files = os.listdir(path)
    print 'Preparing to train %s %s files' % (len(files), label)
    for filename in files:
        with open(os.path.join(path, filename)) as fh:
            contents = fh.read()

        # extract the words from the document
        features = extract_features(contents)

        # train the classifier to associate the features with the label
        classifier.train(features, label)

    print 'Trained %s files' % len(files)

def extract_features(s, min_len=2, max_len=20):
    """
    Extract all the words in the string `s` that have a length within
    the specified bounds
    """
    words = []
    for w in s.lower().split():
        wlen = len(w)
        if wlen > min_len and wlen < max_len:
            words.append(w)
    return words


def test(classifier, corpus='corpus2'):
    curdir = os.path.dirname(__file__)

    # paths to spam and ham documents
    spam_dir = os.path.join(curdir, corpus, 'spam')
    ham_dir = os.path.join(curdir, corpus, 'ham')

    correct = total = 0

    for path, label in ((spam_dir, 'spam'), (ham_dir, 'ham')):
        filenames = os.listdir(path)
        print 'Preparing to test %s %s files from %s.' % (
            len(filenames),
            label,
            corpus)

        for filename in os.listdir(path):
            with open(os.path.join(path, filename)) as fh:
                contents = fh.read()

            # extract the words from the document
            features = extract_features(contents)

            results = classifier.classify(features)

            if results[0][0] == label:
                correct += 1
            total += 1

    pct = 100 * (float(correct) / total)
    print '[%s]: processed %s documents, %02f%% accurate' % (corpus, total, pct)

if __name__ == '__main__':
    classifier = train()
    test(classifier, 'corpus2')
    test(classifier, 'corpus3')
    classifier.close()
    os.unlink('enron.kct')
