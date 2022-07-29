# Linear Regression

# Model

def predict(X,w):
  assert X.shape[-1] == w.shape[0]
  return X@w

# Loss

def loss(X,w,y):
  e = predict(X,w) - y
  return 0.5 * ( np.transpose(e) @ e )
