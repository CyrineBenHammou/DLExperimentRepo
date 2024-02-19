import numpy as np

# Reading Data
data = open('Eminem.txt', 'r').read()

# Initializing Parameters
chars = list(set(data))
seq_length = 25
learning_rate = 0.3
data_size, vocab_size = len(data), len(chars)

# Printing Data Statistics
print('data has %d chars, %d unique' % (data_size, vocab_size))

# Initializing Neural Network Parameters
output_size = vocab_size
num_hidden_units = 50
iteration_count = 300000

# Creating Character Index Dictionaries
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
print(char_to_ix)
print(ix_to_char)


combined = num_hidden_units + vocab_size

# Initializing Neural Network Weights and Biases

# Weight matrices
wa = np.random.rand(num_hidden_units, combined) / np.sqrt(combined/2.)
wf = np.random.rand(num_hidden_units, combined) / np.sqrt(combined/2.)
wo = np.random.rand(num_hidden_units, combined) / np.sqrt(combined/2.)
wi = np.random.rand(num_hidden_units, combined) / np.sqrt(combined/2.)
wy = np.random.rand(output_size, num_hidden_units) / np.sqrt(num_hidden_units/2.)

# Bias vectors
bf = np.zeros(num_hidden_units)
bo = np.zeros(num_hidden_units)
ba = np.zeros(num_hidden_units)
bi = np.zeros(num_hidden_units)
by = np.zeros(output_size)


# Defining Activation Functions and Loss Functions
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Multiclass loss function
def loss_multi(y, t):
    return np.sum(1 - y[t])

# Squared loss function
def loss(pred, label):
    return (pred[0][0] - label) ** 2

# Derivative of the sigmoid function
def sigmoid_derivative(values):
    return values * (1 - values)

# Derivative of the hyperbolic tangent function.
def tanh_derivative(values):
    return 1 - values ** 2

# Alternative derivative of the hyperbolic tangent function
def tanh_derivative2(values):
    return 1 - (np.tanh(values)) ** 2

# Softmax function.
def soft_max(values):
    return np.exp(values) / np.sum(np.exp(values))


#Feedforward + Backpropagation
def loss_function(inputs, hp, sp, y):

    # Initialize dictionaries to store gradients and intermediate values
    g_s = {}
    g_f = {}
    g_o = {}
    g_i = {}
    g_a = {}
    g_y = {}
    x = {}
    h = {}

    # Initialize previous hidden state and LSTM state
    h[-1] = np.atleast_2d(hp)
    g_s[-1] = np.atleast_2d(sp)

    # Initialize loss value
    loss = 0

    # Convert true labels to numpy array
    y = np.asarray(y)

    # Forward pass through time steps
    for t in range(len(y)):

        # One-hot encode input character
        x[t] = np.atleast_2d(np.zeros((vocab_size)))
        x[t][0, inputs[t]] += 1
        # Concatenate input with previous hidden state
        x[t] = np.hstack((x[t], h[t - 1]))[0]

        # Calculate gate activations
        g_f[t] = sigmoid(bf + np.dot(wf, x[t]))
        g_i[t] = sigmoid(bi + np.dot(wi, x[t]))
        g_a[t] = np.tanh(ba + np.dot(wa, x[t]))
        g_o[t] = sigmoid(bo + np.dot(wo, x[t]))

        # Calculate LSTM state
        g_s[t] = g_f[t] * g_s[t - 1] + g_i[t] * g_a[t]

        # Calculate hidden state
        h[t] = np.tanh(g_s[t]) * g_o[t]

        # Calculate output probabilities
        g_y[t] = soft_max(np.dot(wy, h[t][0]) + by)

        # Update loss
        loss += loss_multi(g_y[t], y[t])

    # Initialize gradients for weights and biases
    duo, dbo = np.zeros_like(wo), np.zeros_like(bo)
    dua, dba = np.zeros_like(wa), np.zeros_like(ba)
    duf, dbf = np.zeros_like(wf), np.zeros_like(bf)
    dui, dbi = np.zeros_like(wi), np.zeros_like(bi)
    duy, dby = np.zeros_like(wy), np.zeros_like(by)
    delta_out = np.zeros(num_hidden_units)
    delta_s_future = np.zeros(num_hidden_units)

    # Backpropagation through time steps
    for t in reversed(range(len(y))):
        # Calculate prediction error
        pred = g_y[t]
        label = np.zeros_like(pred)
        label[y[t]] = 1
        diff = 2 * (pred - label)

        # Update gradients for output layer
        L = diff
        L = np.atleast_2d(L)

        # Calculate gradients for LSTM gates
        duy += np.dot(L.T, h[t])
        dby += L[0]
        dout = np.dot(L, wy) + delta_out
        dstate = dout * g_o[t] * tanh_derivative2(g_s[t]) + delta_s_future
        df = dstate * g_s[t - 1] * sigmoid_derivative(g_f[t])
        di = dstate * g_a[t] * sigmoid_derivative(g_i[t])
        da = dstate * g_i[t] * tanh_derivative(g_a[t])
        do = dout * np.tanh(g_s[t]) * sigmoid_derivative(g_o[t])

        # Calculate delta_out for the next time step
        delta_out = np.dot(wi.T, di[0]) \
                   + np.dot(wo.T, do[0]) \
                   + np.dot(wf.T, df[0]) \
                   + np.dot(wa.T, da[0])

        delta_out = delta_out[vocab_size:]

        # Calculate delta_s_future for the next time step
        delta_s_future = dstate * g_f[t]

        # Update gradients for weights and biases
        duo += np.outer(do, x[t])
        dui += np.outer(di, x[t])
        duf += np.outer(df, x[t])
        dua += np.outer(da, x[t])

        dbo += do[0]
        dba += da[0]
        dbf += df[0]
        dbi += di[0]

    #Gradient clipping
    for dparam in [duf, duo, dui, dua, duy, dbo, dbi, dba, dbf, dby]:
        np.clip(dparam, -1, 1, out=dparam)

    # Return loss and gradients
    return loss, duf, duo, dui, dua, duy, dbo, dbi, dba, dbf, dby, h[len(y) - 1], g_s[len(y) - 1], g_y


def sample(inputs, hp, sp, n):
    ixes = []
    # Initialize input vector with zeros and set the input character to 1
    x = np.atleast_2d(np.zeros(vocab_size))[0]
    x[inputs] += 1

    for t in range(n):
        # Concatenate input vector with previous hidden state
        x = np.hstack((x, hp[0]))
        # Calculate forget gate activation
        Gf = sigmoid(bf + np.dot(wf, x))
        # Calculate input gate activation
        Gi = sigmoid(bi + np.dot(wi, x))
        # Calculate candidate activation
        Ga = np.tanh(ba + np.dot(wa, x))
        # Calculate output gate activation
        Go = sigmoid(bo + np.dot(wo, x))
        # Calculate LSTM state
        Gs = Gf * sp + Gi * Ga
        # Calculate hidden state
        h = np.tanh(Gs) * Go 
        # Calculate output probabilities using softmax
        Gy = soft_max(np.dot(wy, h[0]) + by)
        # Sample the next character index based on probabilities
        ix = np.random.choice(range(vocab_size), p=Gy.ravel())
        # Reset input vector
        x = np.atleast_2d(np.zeros(vocab_size))[0]
        # Set the sampled character to 1 in the input vector
        x[ix] += 1
        # Append sampled index to the list
        ixes.append(ix)
        # Update hidden state for the next iteration
        hp = h
        # Update LSTM state for the next iteration
        sp = Gs
    # Convert sampled indices to characters
    txt = ''.join(ix_to_char[ix] for ix in ixes)
    # Print the generated text
    print('----\n %s \n----' % (txt,))


def train():

    # Initialize iteration and data pointer
    n, p = 0, 0
    # Initialize previous hidden state
    hprev = np.zeros((num_hidden_units, 1))
    # Initialize memory variables for AdaGrad
    mdUf, mdUo, mdUi, mdUs, mdUy = np.zeros_like(wf), np.zeros_like(wo), np.zeros_like(wi), np.zeros_like(
        wa), np.zeros_like(wy)
    # Initialize memory variables for AdaGrad
    mdbo, mdbi, mdbs, mdbf, mdby = np.zeros_like(bo), np.zeros_like(bi), np.zeros_like(ba), np.zeros_like(
        bf), np.zeros_like(by)

    # Set random seed for reproducibility
    np.random.seed(0)

    while n <= iteration_count:
        # Reset LSTM memory and data pointer if necessary
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros(num_hidden_units)  # reset LSTM memory
            sprev = np.zeros(num_hidden_units)
            p = 0  # go from start of data

        # Prepare inputs and targets for the current sequence
        inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

        # Calculate loss and gradients using the loss function
        loss, dUf, dUo, dUi, dUs, dUy, dbo, dbi, dbs, dbf, dby, hprev, sprev, rez = loss_function(inputs, hprev,
                                                                                             sprev, targets)

        # Print loss and sample from the model periodically
        if n % 1000 == 0:
            print('iter %d, loss: %f' % (n, loss))
            sample(inputs[0], hprev, sprev, 200)

        # Adagrad update: update parameters using the calculated gradients
        for param, dparam, mem in zip([wf, wo, wi, wa, wy, bo, bi, ba, bf, by],
                                      [dUf, dUo, dUi, dUs, dUy, dbo, dbi, dbs, dbf, dby],
                                      [mdUf, mdUo, mdUi, mdUs, mdUy, mdbo, mdbi, mdbs, mdbf, mdby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(
                mem + 1e-8)

        p += seq_length  # move data pointer
        n += 1  # iteration counter

# Execute the training process
train()