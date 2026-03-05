 Flow of code in order 


 Text -> Tokenization -> Sequences -> Padding -> Embedding -> RNN -> Dense -> Train -> Predict


# Tokenization

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
print(word_index)

# Convert Text → Sequences

sequences = tokenizer.texts_to_sequences(texts)

print(sequences)

# Padding (Make equal length)

max_len = 5

padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

print(padded_sequences)

# Build Model

Architecture:

Embedding → RNN → Dense

model = Sequential()

model.add(Embedding(input_dim=1000, output_dim=16, input_length=max_len))

model.add(SimpleRNN(32))

model.add(Dense(1, activation='sigmoid'))
 
