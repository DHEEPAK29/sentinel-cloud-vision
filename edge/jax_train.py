import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import cv2

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x) # 10 dummy classes 
        return x

def create_train_state(rng, learning_rate, input_shape):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)

def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        # Dummy loss with random labels for demo
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def run_finetuning(image_bytes, target_object="unknown", steps=5):
    """
    Simulates a finetuning loop on the provided image/dataset.
    """
    try:
        # Preprocess image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
             return {"status": "error", "message": "Could not decode image"}
        
        img = cv2.resize(img, (28, 28)) # Resize to small fixed size for demo CNN
        img = cv2.resize(img, (28, 28)) # Resize to small fixed size for demo CNN
        img = img / 255.0
        img = np.expand_dims(img, axis=0) # Add batch dim
        
        # Initialize
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        state = create_train_state(init_rng, 1e-3, [1, 28, 28, 3])
        
        # Dummy "Training" loop
        losses = []
        for i in range(steps):
            # Create dummy batch (single image, random label)
            batch = {
                'image': jnp.array(img),
                'label': jax.random.randint(rng, (1,), 0, 10)
            }
            state, loss = train_step(state, batch)
            losses.append(float(loss))
            
        return {
            "status": "success", 
            "final_loss": losses[-1], 
            "steps": steps,
            "params": state.params  # Return the trained parameters
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_inference(image_bytes):
    """
    Performs JAX inference on the provided image.
    """
    try:
        # Preprocess image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
             return {"status": "error", "message": "Could not decode image"}
        
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = np.expand_dims(img, axis=0) # Add batch dim
        
        # Initialize (in a real app, we'd load saved weights)
        rng = jax.random.PRNGKey(0)
        cnn = CNN()
        params = cnn.init(rng, jnp.ones([1, 28, 28, 3]))['params']
        
        # Inference
        logits = cnn.apply({'params': params}, jnp.array(img))
        probs = jax.nn.softmax(logits)
        predicted_class = int(jnp.argmax(probs))
        confidence = float(jnp.max(probs))
        
        return {
            "status": "success",
            "class_id": predicted_class,
            "confidence": confidence,
            "backend": "jax/flax"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
