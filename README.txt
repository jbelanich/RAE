--- Word Database ---

First I need something that represents a "vocabulary", some set of words
V.  Associated with this vocabulary is the vector representation of each
word, a 100 dimensional vector where each component is sampled from a
normal distribution.  We stack these vectors into a 100 by |V| matrix where
the kth column corresponds to the kth word in V.

What is the best way to store this efficiently?  How about a database mapping
words to vectors.  The primary thing I need this for is for lookup...I need to
be able to quickly grab the vector associated with a word.

Note that whenever I encounter a new word, I need to insert it into the
database.

--- Training Set ---

Twitter is an excellent source of unlabeled data, but I need labeled data for
sentiment analysis.  I might be able to create a good dataset from twitter manually.

The movie review dataset might also work for labeled data.

--- Getting Vector Representation ---
Given a sentence S = [w_1, w_2, ..., w_m], I want to transform it into a list
of vectors [x_1, x_2, ..., x_m].

transformed_sentence = []
for word in S:
	transformed_sentence.append(search_for_vector(word))

--- Getting the output of the RAE ---
This will describe at a high level the procedure for RNE(S, W, b), for some sentence
S, weight matrix W, and bias b.  This will return the following tuple (T, reconstruction error) where T is the tree constructed.  T has a number of parameters
at each layer node indexed s: (reconstruction_error_s, p_s, c_s), where p_s is the
parent representation, c_s is the classification of the node (bad or good sentiment).

computing p_i = W^(1)*[x_i-1;x_i] + b for each i and selecting p that minimizes
reconstruction error epsilon_i.  Suppose that p_k is the best one, we then say that
S_2 = [x_1,x_2,...,p_k,...,x_m] where p_k replaces x_k-1 and x_k.  But we remember p_s = p_k, epsilon_s = epsilon_k, c_s = softmax(W^(label)*p_s) and remember the structure of the tree somehow (todo).

--- Training ---
Not quite sure...I need to read backprop through structure paper.

--- Matrices ---
How do I do matrices in python?  Scipy?