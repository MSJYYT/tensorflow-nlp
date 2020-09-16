		Code has been run on Google Colab, thanks Google for providing computational resources

#### Contents

* Natural Language Processing（自然语言处理）

	* [Text Classification（文本分类）](https://github.com/zhedongzheng/finch#text-classification)
	
		* IMDB（English Data）

			```
			Abstract:
			
			1. we show the classic ml model (tfidf + logistic regression) is able to reach 89.6%

			   which is decent for its simplicity, efficiency and low-cost

			2. we show fasttext model is able to reach 90% accuracy

			3. we show cnn-based model is able to improve the accuracy to 91.7%

			4. we show rnn-based model is able to improve the accuracy to 92.6%

			5. we show pretrained model (bert) is able to improve the accuracy to 94%

			6. we show pretrained model (roberta) is able to improve the accuracy to 94.7%

			7. we use back-translation, label smoothing, cyclical lr as training helpers
			```

	* [Text Matching（文本匹配）](https://github.com/zhedongzheng/finch#text-matching)
	
		* SNLI（English Data）

			```
			Abstract:

			1. we show dam (lots of interact) is able to reach 85.3% accuracy

			2. we show pyramid (rnn + image processing) is able to improve the accuracy to 87.1%

			3. we show esim (rnn + lots of interact) is able to improve the accuracy to 87.4%

			4. we show re2 (rnn + lots of interact + residual) is able to improve the accuracy to 88.3%

			5. we show bert (pretrained model) is able to improve the accuracy to 90.4%

			6. we show roberta (pretrained model) is able to improve the accuracy to 91.1%

			7. we use label smoothing and cyclical lr as training helpers
			```

		* 微众银行智能客服（Chinese Data）

			```
			Abstract:

			1. we show esim, pyramid, re2 are able to reach 82.5% ~ 82.9% accuracy (very close)

			2. we show re2 is able to be improved to 83.8% by using cyclical lr and label smoothing

			3. we show bert (pretrained model) is able to further improve the accuracy to 84.75%

			4. we show char-level re2 actually performs better than word-level re2 on this dataset
			```

	* [Spoken Language Understanding（对话理解）](https://github.com/zhedongzheng/finch#spoken-language-understanding)

		* ATIS（English Data）

			```
			Abstract:

			1. we show the baseline crf is able to reach 92% accuracy

			2. we show birnn can reach 95.8% micro-f1 for slots and 97.2% accuracy for intents

			3. we show native transformer can reach 95.5% micro-f1 for slots and 96.5% accuracy for intents

			   after applying time-mixing, it can reach 95.8% micro-f1 for slots and 97.5% accuracy for intents

			4. we show elmo embedding is effective and can reach 96.3% micro-f1 for slots and 97.3% accuracy for intents
			```

	* [Generative Dialog（生成式对话）](https://github.com/zhedongzheng/finch#generative-dialog)

		* Large-scale Chinese Conversation Dataset

			```
			Abstract:

			1. we show how to train a LSTM based Seq2Seq to generate natural human-like responses in chinese free chat

			2. we show how to add Pointer Net to Seq2Seq, which becomes Pointer-Generator that can generate better responses

			3. we test some powerful GPT models and compare their responses with our LSTM model's
			```

	* [Retrieval Dialog（检索式对话）](https://github.com/zhedongzheng/finch#retrieval-dialog)

		* Sparse Retrieval

		* Dense Retrieval

	* [Multi-turn Dialogue Rewriting（多轮对话改写）](https://github.com/zhedongzheng/finch#multi-turn-dialogue-rewriting)

		* 20k 腾讯 AI 研发数据（Chinese Data）
				
			```
			Highlight:
			
			1. our implementation of rnn-based pointer network reaches 60% exact match without bert

			   which is higher than other implementations using bert

			   e.g. (https://github.com/liu-nlper/dialogue-utterance-rewriter) 57.5% exact match

			2. we show how to deploy model in java production

			3. we explain this task can be decomposed into two stages (extract keywords & recombine query)

			   the first stage is fast (tagging) and the second stage is slow (autoregressive generation)

			   for the first stage, we show birnn extracts keywords at 79.6% recall and 42.6% exact match

			   then we show bert is able to improve this extraction task to 93.6% recall and 71.6% exact match
			
			4. we find that we need to predict the intent as well (whether to rewrite the query or not)

			   in other words, whether to trigger the rewriter or not at the first place

			   we have finetuned a bert to jointly predict intent and extract the keywords
			   
			   the result is: 97.9% intent accuracy; 90.2% recall and 64.3% exact match for keyword extraction
			```

	* [Semantic Parsing（语义解析）](https://github.com/zhedongzheng/finch#semantic-parsing)
	
		* Facebook AI Research Data（English Data）

			```
			Highlight:
			
			our implementation of pointer-generator reaches 80.3% exact match on testing set

			which is higher than all the results of the original paper including rnng (78.5%)

			we further improve exact match to 81.1% by adding more embeddings (char & contextual)

			(https://aclweb.org/anthology/D18-1300)
			```
	
	* [Multi-hop Question Answering（多跳问题回答）](https://github.com/zhedongzheng/finch#multi-hop-question-answering)
	
		* bAbI（Engish Data）
		
	* [Text Processing Tools（文本处理工具）](https://github.com/zhedongzheng/finch#text-processing-tools)

* Knowledge Graph（知识图谱）

	* [Knowledge Graph Inference（知识图谱推理）](https://github.com/zhedongzheng/finch#knowledge-graph-inference)
	
	* [Knowledge Base Question Answering（知识图谱问答）](https://github.com/zhedongzheng/finch#knowledge-base-question-answering)
	
	* [Knowledge Graph Tools（知识图谱工具）](https://github.com/zhedongzheng/finch#knowledge-graph-tools)

* [Recommender System（推荐系统）](https://github.com/zhedongzheng/finch#recommender-system)

	* Movielens 1M（English Data）

---

## Text Classification

```
└── finch/tensorflow2/text_classification/imdb
	│
	├── data
	│   └── glove.840B.300d.txt          # pretrained embedding, download and put here
	│   └── make_data.ipynb              # step 1. make data and vocab: train.txt, test.txt, word.txt
	│   └── train.txt  		     # incomplete sample, format <label, text> separated by \t 
	│   └── test.txt   		     # incomplete sample, format <label, text> separated by \t
	│   └── train_bt_part1.txt  	     # (back-translated) incomplete sample, format <label, text> separated by \t
	│
	├── vocab
	│   └── word.txt                     # incomplete sample, list of words in vocabulary
	│	
	└── main              
		└── attention_linear.ipynb   # step 2: train and evaluate model
		└── attention_conv.ipynb     # step 2: train and evaluate model
		└── fasttext_unigram.ipynb   # step 2: train and evaluate model
		└── fasttext_bigram.ipynb    # step 2: train and evaluate model
		└── sliced_rnn.ipynb         # step 2: train and evaluate model
		└── sliced_rnn_bt.ipynb      # step 2: train and evaluate model
```

* Task: [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)（English Data）
	
        Training Data: 25000, Testing Data: 25000, Labels: 2
	
	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/train.txt)
		
		* [\<Text File>: Data Example (Back-Translated)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/train_bt_part1.txt)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/vocab/word.txt)

	* Model: TF-IDF + Logistic Regression
	
		* PySpark
		
			* [\<Notebook> Unigram + TF + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/spark/text_classification/imdb/tfidf_lr.ipynb)
			
			 	-> 88.2% Testing Accuracy
			
		* Sklearn
		
			* [\<Notebook> Unigram + TF + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_false.ipynb)
			
			 	-> 88.3% Testing Accuracy
			
			* [\<Notebook> Unigram + TF (binary) + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_true.ipynb)
			
			 	-> 88.8% Testing Accuracy

			* [\<Notebook> Unigram + Bigram + TF (binary) + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_true_bigram.ipynb)
			
			 	-> 89.6% Testing Accuracy

	* Model: [FastText](https://arxiv.org/abs/1607.01759)
	
		* [Facebook Official Release](https://github.com/facebookresearch/fastText)
		
			* [\<Notebook> Unigram FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/unigram.ipynb)
		
		 		-> 87.3% Testing Accuracy
		
			* [\<Notebook> (Unigram + Bigram) FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/bigram.ipynb)

				-> 89.8% Testing Accuracy

			* [\<Notebook> Auto-tune FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/autotune.ipynb)

				-> 90.1% Testing Accuracy

		* TensorFlow 2

			* [\<Notebook> Unigram FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/fasttext_unigram.ipynb)
				
			 	-> 89.1 % Testing Accuracy
				
			* [\<Notebook> (Unigram + Bigram) FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/fasttext_bigram.ipynb)
	
	 			-> 90.2 % Testing Accuracy
	
	* Model: [Feedforward Attention](https://arxiv.org/abs/1512.08756)

		* TensorFlow 2

			* [\<Notebook> Feedforward Attention](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/attention_linear.ipynb)
			
			 	-> 89.5 % Testing Accuracy
			
			* [\<Notebook> CNN + Feedforward Attention](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/attention_conv.ipynb)

				-> 90.7 % Testing Accuracy

			* [\<Notebook> CNN + Feedforward Attention + Back-Translation + Char Embedding + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/cnn_attention_bt_char_label_smooth_cyclical.ipynb)
			
				-> 91.7 % Testing Accuracy

	* Model: [Sliced RNN](https://arxiv.org/abs/1807.02291)

		* TensorFlow 2

			* [\<Notebook> Sliced LSTM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn.ipynb)

 				-> 91.4 % Testing Accuracy

			* [\<Notebook> Sliced LSTM + Back-Translation](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt.ipynb)

 				-> 91.7 % Testing Accuracy
				
				```
				Back-Translation increases training data from 25000 to 50000

				which is done by "english -> french -> english" translation
				```

				```python
				from googletrans import Translator

				translator = Translator()

				translated = translator.translate(text, src='en', dest='fr').text
				
      			back = translator.translate(translated, src='fr', dest='en').text
				```

			* [\<Notebook> Sliced LSTM + Back-Translation + Char Embedding](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt_char.ipynb)

 				-> 92.3 % Testing Accuracy

			* [\<Notebook> Sliced LSTM + Back-Translation + Char Embedding + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt_char_label_smooth.ipynb)
			
				-> 92.5 % Testing Accuracy

			* [\<Notebook> Sliced LSTM + Back-Translation + Char Embedding + Label Smoothing + Cyclical LR](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt_char_label_smooth_clr.ipynb)
			
				-> 92.6 % Testing Accuracy

				This result (without transfer learning) is higher than [CoVe](https://arxiv.org/pdf/1708.00107.pdf) (with transfer learning)

	* Model: [BERT](https://arxiv.org/abs/1810.04805)

		* TensorFlow 2 + [transformers](https://github.com/huggingface/transformers)

			* [\<Notebook> BERT (base-uncased) { batch_size=32, max_len=128 }](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_32_128.ipynb)
			
			 	-> 92.6% Testing Accuracy

			* [\<Notebook> BERT (base-uncased) { batch_size=16, max_len=200 }](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_16_200.ipynb)
			
			 	-> 93.3% Testing Accuracy

			* [\<Notebook> BERT (base-uncased) { batch_size=12, max_len=256 }](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_12_256.ipynb)
			
			 	-> 93.8% Testing Accuracy

			* [\<Notebook> BERT (base-uncased) { batch_size=8, max_len=300 }](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/bert_finetune_8_300.ipynb)
			
			 	-> 94% Testing Accuracy

	* Model: [RoBERTa](https://arxiv.org/abs/1907.11692)

		* TensorFlow 2 + [transformers](https://github.com/huggingface/transformers)

			* [\<Notebook> RoBERTa (base) { batch_size=8, max_len=300 }](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/roberta_finetune_8_300.ipynb)
			
			 	-> 94.7% Testing Accuracy
---

## Text Matching

```
└── finch/tensorflow2/text_matching/snli
	│
	├── data
	│   └── glove.840B.300d.txt       # pretrained embedding, download and put here
	│   └── download_data.ipynb       # step 1. run this to download snli dataset
	│   └── make_data.ipynb           # step 2. run this to generate train.txt, test.txt, word.txt 
	│   └── train.txt  		  # incomplete sample, format <label, text1, text2> separated by \t 
	│   └── test.txt   		  # incomplete sample, format <label, text1, text2> separated by \t
	│
	├── vocab
	│   └── word.txt                  # incomplete sample, list of words in vocabulary
	│	
	└── main              
		└── dam.ipynb      	  # step 3. train and evaluate model
		└── esim.ipynb      	  # step 3. train and evaluate model
		└── ......
```

* Task: [SNLI](https://nlp.stanford.edu/projects/snli/)（English Data）

        Training Data: 550152, Testing Data: 10000, Labels: 3

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/download_data.ipynb)
	
	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/train.txt)
		
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/vocab/word.txt)

	* TensorFlow 2

		* Model: [DAM](https://arxiv.org/abs/1606.01933)
		
			* [\<Notebook> DAM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/dam.ipynb)
			
 				-> 85.3% Testing Accuracy
			
			 	The accuracy of this implementation is higher than [UCL MR Group](http://isabelleaugenstein.github.io/papers/JTR_ACL_demo_paper.pdf)'s implementation (84.6%)

		* Model: [Match Pyramid](https://arxiv.org/abs/1602.06359)
			
			* [\<Notebook> Pyramid](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/pyramid_multi_attn.ipynb)

 				-> 87.1% Testing Accuracy

	 		 	The accuracy of this model is 0.3% below ESIM, however the speed is 1x faster than ESIM

		* Model: [ESIM](https://arxiv.org/abs/1609.06038)
		
			* [\<Notebook> ESIM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/esim.ipynb)

 				-> 87.4% Testing Accuracy

			 	The accuracy of this implementation is comparable to [UCL MR Group](http://isabelleaugenstein.github.io/papers/JTR_ACL_demo_paper.pdf)'s implementation (87.2%)

		* Model: [RE2](https://arxiv.org/abs/1908.00300)
		
			* [\<Notebook> RE2](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_birnn.ipynb)

 				-> 87.7% Testing Accuracy

			* [\<Notebook> RE3](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_3_birnn.ipynb)

 				-> 88.0% Testing Accuracy

			* [\<Notebook> RE3 + Cyclical LR + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_3_birnn_label_smooth.ipynb)

				-> 88.3% Testing Accuracy

		* Model: [BERT](https://arxiv.org/abs/1810.04805)

			* [\<Notebook> BERT (base-uncased)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/bert_finetune.ipynb)
			
				-> 90.4% Testing Accuracy

		* Model: [RoBERTa](https://arxiv.org/abs/1907.11692)

			* [\<Notebook> RoBERTa (base)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/roberta_finetune.ipynb)
			
				-> 91.1% Testing Accuracy

<img src="https://pic2.zhimg.com/80/v2-3ae885000f570573020afa0c4ce65a19_720w.jpg" height="300">

```
└── finch/tensorflow2/text_matching/chinese
	│
	├── data
	│   └── make_data.ipynb           # step 1. run this to generate char.txt and char.npy
	│   └── train.csv  		  # incomplete sample, format <text1, text2, label> separated by comma 
	│   └── test.csv   		  # incomplete sample, format <text1, text2, label> separated by comma
	│
	├── vocab
	│   └── cc.zh.300.vec             # pretrained embedding, download and put here
	│   └── char.txt                  # incomplete sample, list of chinese characters
	│   └── char.npy                  # saved pretrained embedding matrix for this task
	│	
	└── main              
		└── pyramid.ipynb      	  # step 2. train and evaluate model
		└── esim.ipynb      	  # step 2. train and evaluate model
		└── ......
```

* Task: [微众银行智能客服](https://github.com/terrifyzhao/text_matching/tree/master/input)（Chinese Data）

        Training Data: 100000, Testing Data: 10000, Labels: 2

	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/data/train.csv)
		
		* [\<Text File>: Vocabulary](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/vocab/char.txt)
		
	* Model	
	
		* TensorFlow 2
	
			* [\<Notebook> ESIM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/esim.ipynb)

 				-> 82.5% Testing Accuracy

			* [\<Notebook> Pyramid](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/pyramid.ipynb)

 				-> 82.7% Testing Accuracy

			* [\<Notebook> RE2](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/re2.ipynb)

 				-> 82.9% Testing Accuracy

			* [\<Notebook> RE2 + Cyclical LR + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/re2_cyclical_label_smooth.ipynb)

 				-> 83.8% Testing Accuracy

				The result of RE2 actually catches up with Bert base below (both 83.8%)

			These results are higher than [the repo here](https://github.com/terrifyzhao/text_matching) and [the repo here](https://github.com/liuhuanyong/SiameseSentenceSimilarity)

		* TensorFlow 2 + [transformers](https://github.com/huggingface/transformers)

			* [\<Notebook> BERT (chinese_base)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/bert_finetune.ipynb)
			
			 	-> 83.8% Testing Accuracy

		* TensorFlow 1 + [bert4keras](https://github.com/bojone/bert4keras)
		
			* [\<Notebook> BERT (chinese_wwm)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/text_matching/chinese/main/bert_chinese_wwm.ipynb)
			
			 	-> 84.75% Testing Accuracy
			
				Weights downloaded from [here](https://github.com/ymcui/Chinese-BERT-wwm)

	* About word-level vs character-level

		* All these models above have been implemented on character-level

		* We have attempted word-level modelling by using [jieba](https://github.com/fxsjy/jieba) to split words

		* but [word-level RE2](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/word_re2_cyclical_label_smooth.ipynb) (82.5% accuracy) does not surpass char-level RE2 above (83.8%)

---

## Spoken Language Understanding

```
└── finch/tensorflow2/spoken_language_understanding/atis
	│
	├── data
	│   └── glove.840B.300d.txt           # pretrained embedding, download and put here
	│   └── make_data.ipynb               # step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── atis.train.w-intent.iob       # incomplete sample, format <text, slot, intent>
	│   └── atis.test.w-intent.iob        # incomplete sample, format <text, slot, intent>
	│
	├── vocab
	│   └── word.txt                      # list of words in vocabulary
	│   └── intent.txt                    # list of intents in vocabulary
	│   └── slot.txt                      # list of slots in vocabulary
	│	
	└── main              
		└── bigru_clr.ipynb               # step 2. train and evaluate model
		└── bigru_self_attn_clr.ipynb     # step 2. train and evaluate model
		└── bigru_clr_crf.ipynb           # step 2. train and evaluate model
```

* Task: [ATIS](https://github.com/yvchen/JointSLU/tree/master/data)（English Data） 

	<img src="https://www.csie.ntu.edu.tw/~yvchen/f105-adl/images/atis.png" width="500">

        Training Data: 4978, Testing Data: 893

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/atis.train.w-intent.iob)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/vocab/word.txt)

	* Model: [Conditional Random Fields](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
	
		* [CRFsuite](http://www.chokkan.org/software/crfsuite/) + [pycrfsuite](https://github.com/scrapinghub/python-crfsuite)

			* [\<Notebook> CRF](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/python/atis/main/crfsuite.ipynb) 
			
			  92% Slot Micro-F1 on Testing Data

	* Model: [Bi-directional RNN](https://www.ijcai.org/Proceedings/16/Papers/425.pdf)

		* TensorFlow 2

			* [\<Notebook> Bi-GRU](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru_clr.ipynb) 
			
			  97.4% Intent Acc, 95.4% Slot Micro-F1 on Testing Data

			* [\<Notebook> Bi-GRU + CRF](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru_clr_crf.ipynb) 
			
			  97.2% Intent Acc, 95.8% Slot Micro-F1 on Testing Data

	* Model: [Transformer](https://arxiv.org/abs/1706.03762)

		* TensorFlow 2

			* [\<Notebook> 2-layer Transformer](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/transformer.ipynb) 
			
			  96.5% Intent Acc, 95.5% Slot Micro-F1 on Testing Data

			* [\<Notebook> 2-layer Transformer + Time-weighting](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/transformer_time_weight.ipynb) 
			
			  97.2% Intent Acc, 95.6% Slot Micro-F1 on Testing Data

			* [\<Notebook> 2-layer Transformer + Time-mixing](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/transformer_time_mixing.ipynb) 
			
			  97.5% Intent Acc, 95.8% Slot Micro-F1 on Testing Data
			  
		Time-weighting and Time-mixing strategies are borrowed from [this repo](https://github.com/BlinkDL/minGPT-tuned), which are proved to be effective here

	* Model: [ELMO Embedding](https://arxiv.org/abs/1802.05365)
	
		* TensorFlow 1

			* [\<Notebook> ELMO + Bi-GRU](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/spoken_language_understanding/atis/main/elmo_o1_bigru.ipynb) 
			
			  97.5% Intent Acc, 96.1% Slot Micro-F1 on Testing Data

			* [\<Notebook> ELMO + Bi-GRU + CRF](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/spoken_language_understanding/atis/main/elmo_o1_bigru_crf.ipynb) 
			
			  97.3% Intent Acc, 96.3% Slot Micro-F1 on Testing Data

---

## Generative Dialog

```
└── finch/tensorflow1/free_chat/chinese_lccc
	│
	├── data
	│   └── LCCC-base.json           	# raw data downloaded from external
	│   └── LCCC-base_test.json         # raw data downloaded from external
	│   └── make_data.ipynb           	# step 1. run this to generate vocab {char.txt} and data {train.txt & test.txt}
	│   └── train.txt           		# processed text file generated by {make_data.ipynb}
	│   └── test.txt           			# processed text file generated by {make_data.ipynb}
	│
	├── vocab
	│   └── char.txt                	# list of chars in vocabulary for chinese
	│   └── cc.zh.300.vec			# fastText pretrained embedding downloaded from external
	│   └── char.npy			# chinese characters and their embedding values (300 dim)	
	│	
	└── main
		└── lstm_seq2seq_train.ipynb    # step 2. train and evaluate model
		└── lstm_seq2seq_export.ipynb   # step 3. export model
		└── lstm_seq2seq_infer.ipynb    # step 4. model inference
		└── transformer_train.ipynb     # step 2. train and evaluate model
		└── transformer_export.ipynb    # step 3. export model
		└── transformer_infer.ipynb     # step 4. model inference
```

* Task: [Large-scale Chinese Conversation Dataset](https://github.com/thu-coai/CDial-GPT)

        Training Data: 5000000 (sampled due to small memory), Testing Data: 19008
	
	* Data
		
		* [\<Text File>: Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/data/train.txt)

		* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/free_chat/chinese_lccc/data/make_data.ipynb)

			* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/free_chat/chinese_lccc/vocab/char.txt)

	* Model: [Transformer](https://arxiv.org/abs/1706.03762)

		* TensorFlow 1 + [texar](https://github.com/asyml/texar)
			
			* [\<Notebook> Training](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/transformer_train.ipynb)
			
				Transformer Encoder + LSTM Decoder -> 42.465 Testing Perplexity

		* Model Inference
		
			* [\<Notebook> Model Export](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/transformer_export.ipynb)
			
			* [\<Notebook> Python Inference](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/transformer_infer.ipynb)

	* Model: [RNN Seq2Seq + Attention](https://arxiv.org/abs/1409.0473)

		* TensorFlow 1
			
			* [\<Notebook> Training](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_seq2seq_train.ipynb)
			
				LSTM Encoder + LSTM Decoder -> 41.250 Testing Perplexity

		* Model Inference
		
			* [\<Notebook> Model Export](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_seq2seq_export.ipynb)
			
			* [\<Notebook> Python Inference](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_seq2seq_infer.ipynb)

	* Model: [RNN Seq2Seq + Attention + Pointer-Generator](https://arxiv.org/abs/1704.04368)

		* TensorFlow 1
			
			* [\<Notebook> Training](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_pointer_train.ipynb)
			
				LSTM Encoder + LSTM Decoder + Pointer-Generator -> 36.525 Testing Perplexity

		* Model Inference
		
			* [\<Notebook> Model Export](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_pointer_export.ipynb)
			
			* [\<Notebook> Python Inference](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/lstm_pointer_infer.ipynb)

	* If you want to deploy model in Java production

		```
		└── FreeChatInference
			│
			├── data
			│   └── transformer_export/
			│   └── char.txt
			│   └── libtensorflow-1.14.0.jar
			│   └── tensorflow_jni.dll
			│
			└── src              
				└── ModelInference.java
		```

		* [\<Notebook> Java Inference](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/java/FreeChatInference/src/ModelInference.java)

		* If you don't know the input and output node names in Java, you can call:

			```
			!saved_model_cli show --dir ../model/xxx/1587959473/ --tag_set serve --signature_def serving_default
			```

			which will display the node names:

			```
			The given SavedModel SignatureDef contains the following input(s):
			inputs['history'] tensor_info:
				dtype: DT_INT32
				shape: (-1, -1, -1)
				name: history:0
			inputs['query'] tensor_info:
				dtype: DT_INT32
				shape: (-1, -1)
				name: query:0
			The given SavedModel SignatureDef contains the following output(s):
			outputs['output'] tensor_info:
				dtype: DT_INT32
				shape: (-1, -1)
				name: Decoder/decoder/transpose_1:0
			Method name is: tensorflow/serving/predict
			```

	* Model: [GPT](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)

		* TensorFlow 1 + [bert4keras](https://github.com/bojone/bert4keras)
			
			* [\<Notebook> GPT LCCC base](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/gpt_lccc_base.ipynb)
			
			  This model is pre-trained and released from [Tsinghua University](https://github.com/thu-coai/CDial-GPT)
			
			* [\<Notebook> GPT LCCC large](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/gpt_lccc_large.ipynb)
			
			  This model is pre-trained and released from [Tsinghua University](https://github.com/thu-coai/CDial-GPT)

			* [\<Notebook> NEZHA LM](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_lccc/main/nezha_gpt.ipynb)
			
			  This model is pre-trained and released from [Jianlin Su](https://github.com/bojone/nezha_gpt_dialog)

	* Response Comparsion

		| 问题 / 模型回复 | Our RNN | GPT base | GPT large | Nezha |
		| --- | --- | --- | --- | --- |
		| Q: 你好 | A: 你好 | A: 你好 | A: 你好，你好，我是大好人。 | A: 您好 |
		| Q: 早上好 | A: 早上好 | A: 早上好，好久不见 | A: 早上好！ | A: 早啊早啊 |
		| Q: 晚上好 | A: 晚上好 | A: 你好 | A: 好久不见 | A: 晚上好！ |
		| Q: 再见 | A: 再见 | A: 不见 | A: 你是不是不爱我了 | A: 我要走了 |
		| Q: 好久不见 | A: 好久不见 | A: 是呀，好久不见。 | A: 是啊，好久不见。 | A: 是啊，最近怎么样啊 |
		| Q: 想死你了 | A: 我也是 | A: 我也想你， | A: 我不信 | A: 哈哈哈哈哈 |
		| Q: 谢谢你 | A: 不客气 | A: 不用谢 | A: 不用 | A: 不谢不谢， |
		| Q: 爱你 | A: 爱你 | A: 你是爱我 | A: 我也爱你， | A: 我也爱你，你最美 |
		| Q: 你叫什么名字 | A: 你猜 | A: 我也想知道 | A: 叫你 | A: 你是不是想说我的名字太俗了 |
		| Q: 你几岁了 | A: 问女孩子年龄可不是绅士所为哦 ~ | A: 你猜 | A: 你猜 | A: 我不知道 |
		| Q: 现在几点了 | A: 11点 | A: 你说呢？ | A: 十二点 | A: 我也不知道啊 |
		| Q: 今天天气怎么样 | A: 下雨了 | A: 今天很冷，但是我还是觉得很冷 | A: 很好 | A: 不错 |
		| Q: 我们现在在哪里 | A: 在这里 | A: 我在这里等你呢 | A: 我不知道你在说什么 | A: 在我们宿舍楼下。。。 |
		| Q: 你能给我讲个笑话吗 | A: 不能 | A: 讲个笑话 | A: 讲给谁听 | A: 不能 |
		| Q: 你是男孩还是女孩呀 | A: 我是萌妹纸 | A: 你猜 | A: 你猜呀 | A: 我是萌妹纸 |
		| Q: 你会几种语言呀 | A: 你猜 | A: 不会 | A: 你猜呀~ | A: 只会说一句 |
		| Q: 你能陪我玩吗 | A: 可以啊 | A: 可以 | A: 我可以啊 | A: 你猜 |
		| Q: 说话可以大声一点吗 | A: 可以 | A: 可以的 | A: 不要 | A: 我不 |
		| Q: 天气真好 | A: 是的 | A: 是啊！ | A: 天气好是因为有你 | A: 是啊。 |
		| Q: 天气太糟糕了 | A: 是啊 | A: 是啊 | A: 是啊 | A: 是的 |
		| Q: 下雨了 | A: 是的 | A: 你们那边也下雨了 | A: 是呀 | A: 我也是 |
		| Q: 雨好大 | A: 今天下雨了 | A: 是啊 | A: 是的，好大雨 | A: 对呀，我们还穿短袖 |
		| Q: 我讨厌艳阳天 | A: 我也讨厌艳阳天 | A: 我也是。 | A: 我喜欢阴天 | A: 我喜欢 |
		| Q: 好晒啊 | A: 是啊 | A: 是啊 | A: 是啊！！ | A: 是啊，不是很热的 |
		| Q: 今天好冷 | A: 多穿点衣服 | A: 你们都不来 | A: 你那儿也冷了 | A: 是啊。 |
		| Q: 今天好热 | A: 今天热死了 | A: 热死我了 | A: 对啊 | A: 对啊，我都热疯 |
		| Q: 风好大 | A: 是啊 | A: 风太大了。。。 | A: 你不是在上课嘛 | A: 嗯嗯嗯，风好大，冷 |
		| Q: 雾太大了看不清路 | A: 雾太大了 | A: 哈哈，你这是在逗我笑 | A: 我也看不清 | A: 你是不是傻啊 |
		| Q: 打雷了好可怕 | A: 可怕 | A: 我们这边也是，我们这边下雨了，我们这边也没有下雨。 | A: 是啊 | A: 我们这边下雨了 |
		| Q: 下雪了诶 | A: 下雪了？ | A: 是滴！ | A: 嗯嗯嗯！！好大的雪！！ | A: 下了一天了 |
		| Q: 好烦啊 | A: 我也好烦 | A: 我也是 | A: 我好烦 | A: 我也是 |
		| Q: 好开心 | A: 开心就好 | A: 你是不是想我了 | A: 是呀 | A: 哈哈哈，我也是！ |
		| Q: 太激动了 | A: 激动什么 | A: 我也激动 | A: 哈哈，还是不要了 | A: 哈哈 |
		| Q: 我好难过 | A: 怎么了？ | A: 我也是，我们都是 | A: 不要难过，我陪你 | A: 你怎么难过 |
		| Q: 我想哭 | A: 抱抱 | A: 抱紧你 | A: 我不想哭！你不要想太多！ | A: 哭吧哭吧，我陪你哭 |
		| Q: 太好笑了 | A: 笑死我了 | A: 你也笑了？ | A: 我也觉得 | A: 哈哈 |
		| Q: 我好伤心 | A: 怎么了 | A: 你伤心什么 | A: 我也是 | A: 你是不是想笑死我！！下次再给你拍 |
		| Q: 心好痛 | A: 抱抱 | A: 你心痛什么 | A: 心碎一地了。 | A: 怎么了 |
		| Q: 好累啊 | A: 加油 | A: 你不会还是一个人在寝室吧 | A: 好好休息 | A: 累了就睡觉 |
		| Q: 我好疲惫 | A: 我也是 | A: 我也差不多。。 | A: 我好困 | A: 你不是一直都很疲惫么 |
		| Q: 我爱你 | A: 我也爱你 | A: 我也是 | A: 我们都爱你 | A: 爱我就来见面 |
		| Q: 我讨厌你 | A: 我也讨厌你 | A: 你不是一直很喜欢我 | A: 不是我的 | A: 我也讨厌你 |
		| Q: 你真是太棒啦 | A: 谢谢 | A: 你最棒 | A: 你也棒棒哒 | A: 谢谢！ |
		| Q: 你好厉害啊 | A: 哈哈 | A: 是不是 | A: 你也不差呀 | A: 你也可以的 |
		| Q: 吓死我了 | A: 哈哈 | A: 我都不敢出门了 | A: 哈哈哈哈哈！！！ | A: 哈哈哈哈哈笑死我了 |
		| Q: 我想回家 | A: 我也想回家 | A: 我也想！！ | A: 想家的时候回来，想家的时候离开。 | A: 回来吧，家真的好啊！ |
		| Q: 我想爸妈了 | A: 我也想爸妈 | A: 哈哈 | A: 我也想 | A: 想我吗 |
		| Q: 不知道小孩在家有没有听话 | A: 我也不知道 | A: 没有 | A: 听话的话肯定是会听话的。 | A: 我也是听不懂啊 |
		| Q: 想回家撸猫 | A: 我也想回家 | A: 你也想啊？ | A: 我们这也有一个 | A: 回呀回呀 |

---

## Retrieval Dialog

* Task: Build a chatbot answering fundamental questions

	* Engine: Elasticsearch

		* [\<Notebook> Sparse (Default) Retrieval](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/es/free_chat/main/default_retrieve.ipynb)

		* [\<Notebook> Dense Retrieval](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/es/free_chat/main/dense_retrieve.ipynb)

		* Badcase Analysis

			| 问题 / 模型回复 | Sparse Retrieval | Dense Retrieval |
			| --- | --- | --- |
			| Q: 我喜欢你 | A: 书是人类进步的阶梯 | A: 我也爱你 |
			| Q: 热死了 | A: 我也想死你了 | A: 对啊, 热死人了 |

			As can be seen here, sparse retrieval is easy to be attacked by character-level difference

			However, Dense retrieval is more robust to capture the contextual meaning

---

## Semantic Parsing

<img src="https://pic3.zhimg.com/v2-fa2cdccee8c725af42564b37741ba47a_b.jpg">

```
└── finch/tensorflow2/semantic_parsing/tree_slu
	│
	├── data
	│   └── glove.840B.300d.txt     	# pretrained embedding, download and put here
	│   └── make_data.ipynb           	# step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── train.tsv   		  	# incomplete sample, format <text, tokenized_text, tree>
	│   └── test.tsv    		  	# incomplete sample, format <text, tokenized_text, tree>
	│
	├── vocab
	│   └── source.txt                	# list of words in vocabulary for source (of seq2seq)
	│   └── target.txt                	# list of words in vocabulary for target (of seq2seq)
	│	
	└── main
		└── lstm_seq2seq_tf_addons.ipynb           # step 2. train and evaluate model
		└── ......
		
```

* Task: [Semantic Parsing for Task Oriented Dialog](https://aclweb.org/anthology/D18-1300)（English Data）

        Training Data: 31279, Testing Data: 9042

	* [\<Text File>: Data Example](https://github.com/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/train.tsv)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/vocab/target.txt)

	* TensorFlow 2

		* Model: [RNN Seq2Seq + Attention](https://arxiv.org/abs/1409.0473)

			* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/gru_seq2seq_tf_addons_clr.ipynb) GRU + Seq2Seq + Cyclical LR + Label Smoothing ->
			
			  74.1% Exact Match on Testing Data

			* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/lstm_seq2seq_tf_addons_clr.ipynb) LSTM + Seq2Seq + Cyclical LR + Label Smoothing ->
			
			  74.1% Exact Match on Testing Data

		* Model: [Pointer-Generator](https://arxiv.org/abs/1704.04368)

			* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/gru_pointer_tf_addons_clr.ipynb) GRU + Pointer-Generator + Cyclical LR + Label Smoothing ->
			
			  80.3% Exact Match on Testing Data
			  
			  ```
			  Pointer Generator = Pointer Network + Seq2Seq Network
			  
			  This result is quite strong
			  
			  which beats all the exact match results in the original paper
			  
			  (https://aclweb.org/anthology/D18-1300)
			  ```

			* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/gru_pointer_tf_addons_clr_bert_char.ipynb) Char Embedding + Bert Embedding + GRU + Pointer-Generator + Cyclical LR + Label Smoothing ->
			
			  81.1% Exact Match on Testing Data

---

## Knowledge Graph Inference

```
└── finch/tensorflow2/knowledge_graph_completion/wn18
	│
	├── data
	│   └── download_data.ipynb       	# step 1. run this to download wn18 dataset
	│   └── make_data.ipynb           	# step 2. run this to generate vocabulary: entity.txt, relation.txt
	│   └── wn18  		          	# wn18 folder (will be auto created by download_data.ipynb)
	│   	└── train.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│   	└── valid.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t 
	│   	└── test.txt   		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│
	├── vocab
	│   └── entity.txt                  	# incomplete sample, list of entities in vocabulary
	│   └── relation.txt                	# incomplete sample, list of relations in vocabulary
	│	
	└── main              
		└── distmult_1-N.ipynb    	# step 3. train and evaluate model
```

* Task: WN18

        Training Data: 141442, Testing Data: 5000

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/download_data.ipynb)
	
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/wn18/train.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/vocab/relation.txt)
	
	* We use [1-N Fast Evaluation](https://arxiv.org/abs/1707.01476) to largely accelerate evaluation process

		 <img src="https://pic4.zhimg.com/80/v2-8cd8481856f101af45501078b04456bb_720w.jpg">

	* Model: [DistMult](https://arxiv.org/abs/1412.6575)

		* TensorFlow 2

			* [\<Notebook> DistMult -> 79.7% MRR on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/distmult_1-N_clr.ipynb)
	
	* Model: [TuckER](https://arxiv.org/abs/1901.09590)
	
		* TensorFlow 2
		
			* [\<Notebook> TuckER -> 88.5% MRR on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/tucker_1-N_clr.ipynb)
	
	* Model: [ComplEx](https://arxiv.org/abs/1606.06357)

		* TensorFlow 2

			* [\<Notebook> ComplEx -> 93.8% MRR on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/complex_1-N_clr.ipynb)

---

## Knowledge Graph Tools

* Data Scraping

	* [Using Scrapy](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/scrapy/car.ipynb)

	* [Downloaded](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/scrapy/car.csv)

* SPARQL

	* [WN18 Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/rdf_sparql_test.ipynb)

* Neo4j + Cypher

	* [Getting Started](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/neo4j/install_neo4j.ipynb)

---

## Knowledge Base Question Answering

<img src="https://upload-images.jianshu.io/upload_images/17747892-e994edc3518b2d58.png?imageMogr2/auto-orient/strip|imageView2/2/w/880" height="350">

* Rule-based System（基于规则的系统）
	
	For example, we want to answer the following questions:
	
	```
		宝马是什么?  /  what is BMW?
        	我想了解一下宝马  /  i want to know about the BMW
        	给我介绍一下宝马  /  please introduce the BMW to me
		宝马这个牌子的汽车怎么样?  /  how is the car of BMW group?
        	宝马如何呢?  /  how is the BMW?
        	宝马汽车好用吗?  /  is BMW a good car to use?
        	宝马和奔驰比怎么样?  /  how is the BMW compared to the Benz?
        	宝马和奔驰比哪个好?  /  which one is better, the BMW or the Benz?
        	宝马和奔驰比哪个更好?  /  which one is even better, the BMW or the Benz?
	```
	
	* [refo](https://github.com/machinalis/refo) + [jieba](https://github.com/fxsjy/jieba): &nbsp; &nbsp; [\<Notebook> Example](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/python/kbqa/rule_based_qa.ipynb)

---

## Multi-hop Question Answering

<img src="https://github.com/DSKSD/DeepNLP-models-Pytorch/blob/master/images/10.dmn-architecture.png" width='500'>

```
└── finch/tensorflow1/question_answering/babi
	│
	├── data
	│   └── make_data.ipynb           		# step 1. run this to generate vocabulary: word.txt 
	│   └── qa5_three-arg-relations_train.txt       # one complete example of babi dataset
	│   └── qa5_three-arg-relations_test.txt	# one complete example of babi dataset
	│
	├── vocab
	│   └── word.txt                  		# complete list of words in vocabulary
	│	
	└── main              
		└── dmn_train.ipynb
		└── dmn_serve.ipynb
		└── attn_gru_cell.py
```

* Task: [bAbI](https://research.fb.com/downloads/babi/)（English Data）

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/qa5_three-arg-relations_test.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/make_data.ipynb)
	
	* Model: [Dynamic Memory Network](https://arxiv.org/abs/1603.01417)
	
		* TensorFlow 1
		
			* [\<Notebook> DMN -> 99.4% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/main/dmn_train.ipynb)
			
			* [Inference](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/main/dmn_serve.ipynb)

---

## Text Processing Tools

* Word Matching

	* Chinese

		* [\<Notebook>: Regex Rule Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/python/regex/zhcn_extract.ipynb)

* Word Segmentation

	* Chinese
	
		* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/custom_op/tf_jieba.ipynb) Jieba TensorFlow Op purposed by [Junwen Chen](https://github.com/applenob)

* Topic Modelling

	* Data: [2373 Lines of Book Titles](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/spark/topic_modelling/book_titles/all_book_titles.txt)（English Data）

		* Model: TF-IDF + LDA
		
			* PySpark
			
				* [\<Notebook> TF + IDF + LDA](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/spark/topic_modelling/book_titles/lda.ipynb)

			* Sklearn + pyLDAvis
			
				* [\<Notebook> TF + IDF + LDA](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/topic_modelling/book_titles/lda.ipynb)
				
				* [\<Notebook> Visualization](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/topic_modelling/book_titles/lda.html#topic=1&lambda=1&term=)

---

## Recommender System

<img src="https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/image/rec_regression_network.png" width='500'>

```
└── finch/tensorflow1/recommender/movielens
	│
	├── data
	│   └── make_data.ipynb           		# run this to generate vocabulary
	│
	├── vocab
	│   └── user_job.txt
	│   └── user_id.txt
	│   └── user_gender.txt
	│   └── user_age.txt
	│   └── movie_types.txt
	│   └── movie_title.txt
	│   └── movie_id.txt
	│	
	└── main              
		└── dnn_softmax.ipynb
		└── ......
```

* Task: [Movielens 1M](https://grouplens.org/datasets/movielens/1m/)（English Data）
	
        Training Data: 900228, Testing Data: 99981, Users: 6000, Movies: 4000, Rating: 1-5

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/data/make_data.ipynb)

		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/data/train.txt)

	* Model: [Fusion](https://www.paddlepaddle.org.cn/documentation/docs/en/1.5/beginners_guide/basics/recommender_system/index_en.html)
	
		* TensorFlow 1
		
			 > MAE: Mean Absolute Error

			* [\<Notebook> Fusion + Sigmoid ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_sigmoid.ipynb)

 				0.663 Testing MAE

			* [\<Notebook> Fusion + Sigmoid + Cyclical LR ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_sigmoid_clr.ipynb)

 				0.661 Testing MAE

			* [\<Notebook> Fusion + Softmax ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_softmax.ipynb)

 				0.633 Testing MAE

			* [\<Notebook> Fusion + Softmax + Cyclical LR ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_softmax_clr.ipynb)

 				0.628 Testing MAE

				The MAE results seem better than the [all the results here](http://mymedialite.net/examples/datasets.html) and [all the results here](https://test.pypi.org/project/scikit-surprise/)

---

## Multi-turn Dialogue Rewriting

<img src="https://pic1.zhimg.com/80/v2-d80efd57b81c6ece955a247ca7247db4_1440w.jpg" width="600">

```
└── finch/tensorflow1/multi_turn_rewrite/chinese/
	│
	├── data
	│   └── make_data.ipynb         # run this to generate vocab, split train & test data, make pretrained embedding
	│   └── corpus.txt		# original data downloaded from external
	│   └── train_pos.txt		# processed positive training data after {make_data.ipynb}
	│   └── train_neg.txt		# processed negative training data after {make_data.ipynb}
	│   └── test_pos.txt		# processed positive testing data after {make_data.ipynb}
	│   └── test_neg.txt		# processed negative testing data after {make_data.ipynb}
	│
	├── vocab
	│   └── cc.zh.300.vec		# fastText pretrained embedding downloaded from external
	│   └── char.npy		# chinese characters and their embedding values (300 dim)	
	│   └── char.txt		# list of chinese characters used in this project 
	│	
	└── main              
		└── baseline_lstm_train.ipynb
		└── baseline_lstm_export.ipynb
		└── baseline_lstm_predict.ipynb
```

* Task: 20k 腾讯 AI 研发数据（Chinese Data）

	```
	data split as: training data (positive): 18986, testing data (positive): 1008

	Training data = 2 * 18986 because of 1:1 Negative Sampling
	```

	* [\<Text File>: Full Data](https://github.com/chin-gyou/dialogue-utterance-rewriter/blob/master/corpus.txt)
	
	* [\<Notebook>: Make Data & Vocabulary & Pretrained Embedding](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/make_data.ipynb)

			There are six incorrect data and we have deleted them

		* [\<Text File>: Positive Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/train_pos.txt)
		
		* [\<Text File>: Negative Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/train_neg.txt)

	* Model: [RNN Seq2Seq + Attention](https://arxiv.org/abs/1409.0473) + [Multi-hop Memory](https://arxiv.org/abs/1603.01417)

		* TensorFlow 1

			* [\<Notebook> LSTM Seq2Seq + Multi-hop Memory + Attention](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_train_clr.ipynb)

				-> Exact Match: 56.2%, &nbsp; BLEU-1: 94.6, &nbsp; BLEU-2: 89.1, &nbsp; BELU-4: 78.5

			* [\<Notebook> GRU Seq2Seq + Multi-hop Memory + Attention](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_gru_train_clr.ipynb)

				-> Exact Match: 56.6%, &nbsp; BLEU-1: 94.5, &nbsp; BLEU-2: 88.9, &nbsp; BELU-4: 78.3

			* [\<Notebook> GRU Seq2Seq + Multi-hop Memory + Multi-Attention](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_gru_train_clr_multi_attn.ipynb)

				-> Exact Match: 56.2%, &nbsp; BLEU-1: 95.0, &nbsp; BLEU-2: 89.5, &nbsp; BELU-4: 78.9

	* Model: [RNN Pointer Networks](https://arxiv.org/abs/1506.03134)

		* TensorFlow 1

			Pointer Net returns probability distribution, therefore no need to do softmax again in beam search

			Go to beam search [source code](https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/seq2seq/python/ops/beam_search_decoder.py), replace this line

			```python
			step_log_probs = nn_ops.log_softmax(logits)
			```

			with this line

			```python
			step_log_probs = math_ops.log(logits)
			```

			* [\<Notebook> GRU Pointer Net](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/pointer_gru_train_clr.ipynb)

				-> Exact Match: 59.2%, &nbsp; BLEU-1: 93.2, &nbsp; BLEU-2: 87.7, &nbsp; BELU-4: 77.2

			* [\<Notebook> GRU Pointer Net + Multi-Attention](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/pointer_gru_train_clr_multi_attn_.ipynb)

				-> Exact Match: 60.2%, &nbsp; BLEU-1: 94.2, &nbsp; BLEU-2: 88.7, &nbsp; BELU-4: 78.3

				This result (only RNN, without BERT) is comparable to [the result here](https://github.com/liu-nlper/dialogue-utterance-rewriter) with BERT

				```
				Pointer Network is better than Seq2Seq on this kind of task

				where the target text highly overlaps with the source text
				```

	* If you want to deploy model

		* Python Inference（基于 Python 的推理）

			* [\<Notebook> Export](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_export.ipynb)
			
			* [\<Notebook> Inference](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_predict.ipynb)
		
		* Java Inference（基于 Java 的推理）

			* [\<Notebook> Inference](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/java/MultiDialogInference/src/ModelInference.java)

				```
				└── MultiDialogInference
					│
					├── data
					│   └── baseline_lstm_export/
					│   └── char.txt
					│   └── libtensorflow-1.14.0.jar
					│   └── tensorflow_jni.dll
					│
					└── src              
						└── ModelInference.java
				```

	* Despite End-to-End, this problem can also be decomposed into two stages

		* **Stage 1 (Fast). Detecting the (missing or referred) keywords from the context**
		
			which is a sequence tagging task with sequential complexity ```O(1)```

		* Stage 2 (Slow). Recombine the keywords with the query based on language fluency
			
			which is a sequence generation task with sequential complexity ```O(N)```

			```
			For example, for a given query: "买不起" and the context: "成都房价是多少 不买就后悔了成都房价还有上涨空间"

			First retrieve the keyword "成都房" from the context which is very important

			Then recombine the keyword "成都房" with the query "买不起" which becomes "买不起成都房"
			```
		
		* For Stage 1 (sequence tagging for retrieving the keywords), the experiment results are:

			* [\<Notebook> Bi-GRU + Attention](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese_tagging/main/tagging_only_pos.ipynb)
			
				-> Recall: 79.6% &nbsp; Precision: 78.7% &nbsp; Exact Match: 42.6%

			* [\<Notebook> BERT (chinese_base)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/main/bert_finetune.ipynb)
			
				-> Recall: 93.6% &nbsp; Precision: 83.1% &nbsp; Exact Match: 71.6%

	* However, there is still a practical problem to prefine whether the query needs to be rewritten or not

		* if not, we just simply skip the rewriter and pass the query to the next stage

		* there are actually three situations needs to be classified

			* 0: the query does not need to be rewritten because it is irrelevant to the context

				```
				你喜欢五月天吗	超级喜欢阿信	中午出去吃饭吗
				```

			* 1: the query needs to be rewritten

				```
				你喜欢五月天吗	超级喜欢阿信	你喜欢他的那首歌 -> 你喜欢阿信的那首歌
				```

			* 2: the query does not need to be rewritten because it already contains enough information

				```
				你喜欢五月天吗	超级喜欢阿信	你喜欢阿信的那首歌
				```

		* therefore, we aim for training the model to jointly predict:

			* intent: three situations {0, 1, 2} whether the query needs to be rewritten or not

			* keyword extraction: extract the missing or referred keywords in the context

		* [\<Text File>: Positive Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/data/test_pos_tag.txt)
		
		* [\<Text File>: Negative Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/data/test_neg_tag.txt)

		* [\<Notebook> BERT (chinese_base)](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/multi_turn_rewrite/chinese_tagging/main/bert_joint_finetune.ipynb)
			
			-> Intent: 97.9% accuracy

			-> Keyword Extraction: 90.2% recall &nbsp; 80.7% precision &nbsp; 64.3% exact match
