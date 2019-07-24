import operator
import struct
import kyotocabinet as kc


class ClassifierDB(kc.DB):
    """
    Wrapper for `kyotocabinet.DB` that provides utilities for working with
    features and categories.
    """
    def __init__(self, *args, **kwargs):
        super(ClassifierDB, self).__init__(*args, **kwargs)

        self._category_tmpl = 'category.%s'
        self._feature_to_category_tmpl = 'feature2category.%s.%s'
        self._total_count = 'total-count'

    def get_int(self, key):
        # Kyoto serializes ints big-endian 8-bytes long, so we need to unpack
        # them using the `struct` module.
        value = self.get(key)
        if value:
            return struct.unpack('>Q', value)[0]
        return 0

    def incr_feature_category(self, feature, category):
        """Increment the count for the feature in the given category."""
        return self.increment(
            self._feature_to_category_tmpl % (feature, category),
            1)

    def incr_category(self, category):
        """
        Increment the count for the given category, increasing the total
        count as well.
        """
        self.increment(self._total_count, 1)
        return self.increment(self._category_tmpl % category, 1)

    def category_count(self, category):
        """Return the number of documents in the given category."""
        return self.get_int(self._category_tmpl % category)

    def total_count(self):
        """Return the total number of documents overall."""
        return self.get_int(self._total_count)

    def get_feature_category_count(self, feature, category):
        """Get the count of the feature in the given category."""
        return self.get_int(
            self._feature_to_category_tmpl % (feature, category))

    def get_feature_counts(self, feature):
        """Get the total count for the feature across all categories."""
        prefix = self._feature_to_category_tmpl % (feature, '')
        total = 0
        for key in self.match_prefix(prefix):
            total += self.get_int(key)
        return total

    def iter_categories(self):
        """
        Return an iterable that successively yields all the categories
        that have been observed.
        """
        category_prefix = self._category_tmpl % ''
        prefix_len = len(category_prefix)
        for category_key in self.match_prefix(category_prefix):
            yield category_key[prefix_len:]


class NBC(object):
    """
    Simple naive bayes classifier.
    """
    def __init__(self, filename, read_only=False):
        """
        Initialize the classifier by pointing it at a database file. If you
        intend to only use the classifier for classifying documents, specify
        `read_only=True`.
        """
        self.filename = filename
        if not self.filename.endswith('.kct'):
            raise RuntimeError('Database filename must have "kct" extension.')

        self.db = ClassifierDB()
        self.connect(read_only=read_only)

    def connect(self, read_only=False):
        """
        Open the database. Since Kyoto Cabinet only allows a single writer
        at a time, the `connect()` method accepts a parameter allowing the
        database to be opened in read-only mode (supporting multiple readers).
        If you plan on training the classifier, specify `read_only=False`.
        If you plan only on classifying documents, it is safe to specify
        `read_only=True`.
        """
        if read_only:
            flags = kc.DB.OREADER
        else:
            flags = kc.DB.OWRITER
        self.db.open(self.filename, flags | kc.DB.OCREATE)

    def close(self):
        """Close the database."""
        self.db.close()

    def train(self, features, *categories):
        """
        Increment the counts for the features in the given categories.
        """
        for category in categories:
            for feature in features:
                self.db.incr_feature_category(feature, category)
            self.db.incr_category(category)

    def feature_probability(self, feature, category):
        """
        Calculate the probability that a particular feature is associated
        with the given category.
        """
        fcc = self.db.get_feature_category_count(feature, category)
        if fcc:
            category_count = self.db.category_count(category)
            return float(fcc) / category_count
        return 0

    def weighted_probability(self, feature, category, weight=1.0):
        """
        Determine the probability a feature corresponds to the given category.
        The probability is weighted by the importance of the feature, which
        is determined by looking at the feature across all categories in
        which it appears.
        """
        initial_prob = self.feature_probability(feature, category)
        totals = self.db.get_feature_counts(feature)
        return ((weight * 0.5) + (totals * initial_prob)) / (weight + totals)

    def document_probability(self, features, category):
        """
        Calculate the probability that a set of features match the given
        category.
        """
        feature_probabilities = [
            self.weighted_probability(feature, category)
            for feature in features]
        return reduce(operator.mul, feature_probabilities, 1)

    def weighted_document_probability(self, features, category):
        """
        Calculate the probability that a set of features match the given
        category, and weight that score by the importance of the category.
        """
        if self.db.total_count() == 0:
            # Avoid divison by zero.
            return 0

        cat_prob = (float(self.db.category_count(category)) /
                    self.db.total_count())
        doc_prob = self.document_probability(features, category)
        return doc_prob * cat_prob

    def classify(self, features, limit=5):
        """
        Classify the features by finding the categories that match the
        features with the highest probability.
        """
        probabilities = {}
        for category in self.db.iter_categories():
            probabilities[category] = self.weighted_document_probability(
                features,
                category)

        return sorted(
            probabilities.items(),
            key=operator.itemgetter(1),
            reverse=True)[:limit]
