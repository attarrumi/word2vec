package word2vec

import (
	"log"
	"math"
	"strings"
)

const (
// alpha      = 0.025
// iterations = 10
// windowSize = 5
// vectorSize = 10
// minCount   = 5
)

type word struct {
	count  int
	vector []float64
}

type vocabulary struct {
	words      map[string]*word
	alpha      float64
	iterations int
	windowSize int
	vectorSize int
	minCount   int
}

func NewVocabulary(alpha float64, iterations, windowSize, vectorSize, minCount int) *vocabulary {

	return &vocabulary{
		words:      make(map[string]*word),
		alpha:      alpha,
		iterations: iterations,
		windowSize: windowSize,
		vectorSize: vectorSize,
		minCount:   minCount,
	}
}

func (v *vocabulary) AddWord(wordInput string) {
	if _, ok := v.words[wordInput]; !ok {
		v.words[wordInput] = &word{count: 1, vector: make([]float64, v.vectorSize)}
	} else {
		v.words[wordInput].count++
	}
}

func (v *vocabulary) Train(corpusd string) {
	corpus := strings.Split(corpusd, " ")

	for i, w := range corpus {
		if _, ok := v.words[w]; !ok {
			continue
		}
		for j := i - v.windowSize; j <= i+v.windowSize; j++ {
			if j == i || j < 0 || j >= len(corpus) {
				continue
			}
			if _, ok := v.words[corpus[j]]; !ok {
				continue
			}
			v.UpdateWordVectors(v.words[w], v.words[corpus[j]])
		}
	}
}

func (v *vocabulary) UpdateWordVectors(w1, w2 *word) {
	g := (1 - v.alpha) + v.alpha*(float64(w1.count)/float64(v.minCount))
	for i := range w1.vector {
		g1 := v.alpha * (float64(w2.count) * (sigmoid(dotProduct(w1.vector, w2.vector)) - 1) / float64(v.minCount))
		w1.vector[i] -= g1 * g
		w2.vector[i] -= g1 * g
	}
}

func (v *vocabulary) Vector(word string) []float64 {
	wordEmbedding, ok := v.words[word]
	if !ok {
		log.Fatalf("Word '%s' is not in the vocabulary.", word)
	}
	return wordEmbedding.vector
}
func (w2v *vocabulary) GetSimilarWords(word string, topN int) []string {
	wordEmbedding, ok := w2v.words[word]
	if !ok {
		log.Fatalf("Word '%s' is not in the vocabulary.", word)
	}
	similarWords := []string{}
	similarWordsScore := []float64{}
	for w, e := range w2v.words {
		if w == word {
			continue
		}
		score := cosineSimilarity(wordEmbedding.vector, e.vector)
		if len(similarWords) < topN {
			similarWords = append(similarWords, w)
			similarWordsScore = append(similarWordsScore, score)
		} else {
			minScore := math.MaxFloat64
			minIndex := -1
			for i, s := range similarWordsScore {
				if s < minScore {
					minScore = s
					minIndex = i
				}
			}
			if score > minScore {
				similarWords[minIndex] = w
				similarWordsScore[minIndex] = score
			}
		}
	}
	return similarWords
}

func (v *vocabulary) GetSimilarFromVector(vec []float64, topN int) []string {
	similarWords := []string{}
	similarWordsScore := []float64{}
	for w, e := range v.words {

		score := cosineSimilarity(vec, e.vector)
		if len(similarWords) < topN {
			similarWords = append(similarWords, w)
			similarWordsScore = append(similarWordsScore, score)
		} else {
			minScore := math.MaxFloat64
			minIndex := -1
			for i, s := range similarWordsScore {
				if s < minScore {
					minScore = s
					minIndex = i
				}
			}
			if score > minScore {
				similarWords[minIndex] = w
				similarWordsScore[minIndex] = score
			}
		}
	}
	return similarWords
}

// func (v *vocabulary) GetWordFromVector(vector []float64) (string, error) {
// 	var (
// 		minDistance float64 = math.MaxFloat64
// 		word        string
// 	)
// 	for w, e := range v.words {
// 		distance := cosineSimilarity(vector, e.vector)
// 		if distance < minDistance {
// 			minDistance = distance
// 			word = w
// 		}
// 	}
// 	if word == "" {
// 		return "", errors.New("Word not found")
// 	}
// 	return word, nil
// }

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dotProduct(x, y []float64) float64 {
	var sum float64
	for i := range x {
		sum += x[i] * y[i]
	}
	return sum
}

func cosineSimilarity(v1 []float64, v2 []float64) float64 {
	numerator := dotProduct(v1, v2)
	denominator := math.Sqrt(dotProduct(v1, v1) * dotProduct(v2, v2))
	return numerator / denominator
}
