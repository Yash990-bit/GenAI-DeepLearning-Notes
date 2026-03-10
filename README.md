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

 # Compile Model

 model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model

model.fit(
    X_train,
    np.array(y_train),
    epochs=10,
    validation_data=(X_test, y_test)
)

# Evaluate Model

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)

# Predict New Sentence
test_text = ["I love this film"]

seq = tokenizer.texts_to_sequences(test_text)

pad = pad_sequences(seq, maxlen=max_len, padding='post')

prediction = model.predict(pad)

if prediction > 0.5:
    print("Positive")
else:
    print("Negative")

# MLP Code

import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Sample dataset (X: features, y: labels)
# For simplicity, we use a tiny dummy dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],[1],[1],[0]])  # XOR problem

# Hyperparameters
input_size = 2
hidden_size = 4
output_size = 1
lr = 0.5
epochs = 10000

# Initialize weights randomly
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    
    # Compute loss (mean squared error)
    loss = np.mean((y - final_output)**2)
    
    # Backpropagation
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing
hidden_output = sigmoid(np.dot(X, W1) + b1)
final_output = sigmoid(np.dot(hidden_output, W2) + b2)
print("Predictions after training:")
print(final_output)

