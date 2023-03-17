    
	corpus := []string{
		"the quick brown fox jumped over the lazy dog",
		"this is an example of sentence embedding",
		"word2vec is a popular embedding algorithm",
		"go is a popular programming language",
		"machine learning is the future",
	}
	vocab := newVocabulary()
	for _, w := range corpus {
		word := strings.Split(w, " ")
		for _, v := range word {
			vocab.addWord(strings.ToLower(v))
		}
	}

	// Train word vectors
	for i := 0; i < iterations; i++ {
		for _, v := range corpus {
			vocab.train(v)

		}
	}

	// Get word vector
	fmt.Println(vocab)
	d := vocab.GetSimilarWords("lazy", 8)
	fmt.Println(d)