import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# 1. STREAMLIT UI SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="Posture Classifier", layout="centered")
st.title("🤸 Body Language & Posture Classifier")
st.markdown("""
Choose between **CNN** (128×128) and **MobileNetV2** (224×224).  
- **Stop at Target Accuracy** – stops early when validation accuracy reaches a goal.  
- **Calibrate CNN** – reduces overconfidence (only if accuracy is high).  
Both models match the training settings used in the original notebook.
""")

# -------------------------------------------------------------
# 2. DATASET PATH
# -------------------------------------------------------------
DATASET_PATH = "./dataset"

# -------------------------------------------------------------
# 3. CUSTOM CALLBACK: STOP AT TARGET ACCURACY
# -------------------------------------------------------------
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        super().__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get('val_accuracy')
        if current_acc is not None and current_acc >= self.target_accuracy:
            print(f"\n🎯 Target accuracy {self.target_accuracy*100:.1f}% reached at epoch {epoch+1}. Stopping training.")
            self.model.stop_training = True

# -------------------------------------------------------------
# 4. CNN TRAINING (EXACTLY AS NOTEBOOK: 128x128, same architecture)
# -------------------------------------------------------------
def train_cnn(dataset_path, target_accuracy=None):
    tf.random.set_seed(42)
    np.random.seed(42)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(128, 128),
        batch_size=16
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(128, 128),
        batch_size=16
    )
    class_names = train_ds.class_names
    st.session_state["class_names"] = class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
    ])

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("cnn_model.keras", save_best_only=True, monitor='val_accuracy')
    ]
    if target_accuracy is not None:
        callbacks.append(StopAtAccuracy(target_accuracy))

    history = model.fit(train_ds, epochs=25, validation_data=val_ds, callbacks=callbacks, verbose=1)
    return model, history, class_names

# -------------------------------------------------------------
# 5. MOBILENETV2 TRAINING (EXACTLY AS NOTEBOOK: lr=0.001, batch=32, 25 epochs)
# -------------------------------------------------------------
def train_mobilenet(dataset_path, target_accuracy=None):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )
    class_names = train_ds.class_names
    st.session_state["class_names"] = class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.1),
    ])

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("mobilenet_model.keras", save_best_only=True, monitor='val_accuracy')
    ]
    if target_accuracy is not None:
        callbacks.append(StopAtAccuracy(target_accuracy))

    history = model.fit(train_ds, epochs=25, validation_data=val_ds, callbacks=callbacks, verbose=1)
    return model, history, class_names

# -------------------------------------------------------------
# 6. SIDEBAR CONFIGURATION
# -------------------------------------------------------------
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.radio(
    "Select Model Architecture",
    ("CNN", "MobileNetV2")
)

# Stop at target accuracy (both models)
stop_at_target = st.sidebar.checkbox("Stop at target accuracy", value=False)
target_accuracy = None
if stop_at_target:
    target_pct = st.sidebar.slider("Target validation accuracy (%)", 70, 98, 85, 1)
    target_accuracy = target_pct / 100.0
    st.sidebar.info(f"Training will stop when validation accuracy reaches {target_pct}%")

# Calibration (CNN only)
if model_choice == "CNN":
    calibrate = st.sidebar.checkbox("Calibrate CNN confidence (reduce overconfidence)", value=False)
    temperature = st.sidebar.slider("Temperature (higher = lower confidence)", 1.0, 3.0, 1.0, 0.1) if calibrate else 1.0
else:
    calibrate = False
    temperature = 1.0

# -------------------------------------------------------------
# 7. TRAINING BUTTON
# -------------------------------------------------------------
if st.sidebar.button("🚀 Train Model"):
    with st.spinner(f"Training {model_choice}... This may take several minutes."):
        if model_choice == "CNN":
            model, history, class_names = train_cnn(DATASET_PATH, target_accuracy)
            model.save("cnn_model.keras")
            st.session_state["history_cnn"] = history.history
            st.success("✅ CNN model (128×128, notebook settings) trained and saved as 'cnn_model.keras'")
        else:
            model, history, class_names = train_mobilenet(DATASET_PATH, target_accuracy)
            model.save("mobilenet_model.keras")
            st.session_state["history_mobilenet"] = history.history
            st.success("✅ MobileNetV2 model (224×224, notebook settings) trained and saved as 'mobilenet_model.keras'")
        
        if model is None:
            st.error("Training failed. Check dataset path.")
        else:
            # Display training plots
            hist = history.history
            acc_key = 'accuracy' if 'accuracy' in hist else 'acc'
            val_acc_key = f'val_{acc_key}'
            
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(hist[acc_key], label='Train Accuracy', marker='o')
            ax1.plot(hist[val_acc_key], label='Validation Accuracy', marker='o')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(hist['loss'], label='Train Loss', marker='o')
            ax2.plot(hist['val_loss'], label='Validation Loss', marker='o')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Model Loss per Epoch')
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
            
            final_val_acc = hist[val_acc_key][-1] * 100
            st.metric("Validation Accuracy", f"{final_val_acc:.2f}%")
            if stop_at_target and final_val_acc >= target_accuracy*100:
                st.balloons()
                st.success(f"🎯 Target accuracy reached! Training stopped early.")

# -------------------------------------------------------------
# 8. SHOW TRAINING CHARTS (from saved history)
# -------------------------------------------------------------
if st.sidebar.button("📈 Show Training Charts"):
    if model_choice == "CNN" and "history_cnn" in st.session_state:
        hist = st.session_state["history_cnn"]
    elif model_choice == "MobileNetV2" and "history_mobilenet" in st.session_state:
        hist = st.session_state["history_mobilenet"]
    else:
        st.sidebar.warning("No training history found. Train the model first.")
        hist = None
    
    if hist:
        acc_key = 'accuracy' if 'accuracy' in hist else 'acc'
        val_acc_key = f'val_{acc_key}'
        
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(hist[acc_key], label='Train Accuracy', marker='o')
        ax1.plot(hist[val_acc_key], label='Validation Accuracy', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(hist['loss'], label='Train Loss', marker='o')
        ax2.plot(hist['val_loss'], label='Validation Loss', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss per Epoch')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

# -------------------------------------------------------------
# 9. PREDICTION SECTION
# -------------------------------------------------------------
st.subheader("📸 Test Model Prediction")

if model_choice == "CNN":
    model_file = "cnn_model.keras"
else:
    model_file = "mobilenet_model.keras"

if os.path.exists(model_file):
    model = tf.keras.models.load_model(model_file)
    
    # Dynamically get expected input size
    expected_height = model.input_shape[1]
    expected_width = model.input_shape[2]
    st.sidebar.write(f"Model expects {expected_height}×{expected_width} images")
    
    if "class_names" in st.session_state:
        class_names = st.session_state["class_names"]
    else:
        if os.path.exists(DATASET_PATH):
            class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
        else:
            class_names = ['One arm crossed', 'Arms Crossed High (Chest)', 'Arms Crossed Low (Waist)', 
                           'Arms Open and Neutral', 'Hands in Pockets']
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        img = tf.keras.utils.load_img(uploaded_file, target_size=(expected_height, expected_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        
        with st.spinner("Classifying..."):
            logits = model.predict(img_array)
            if calibrate and model_choice == "CNN":
                logits = logits / temperature
                probs = tf.nn.softmax(logits).numpy()[0]
            else:
                probs = logits[0]
            
            predicted_class_idx = np.argmax(probs)
            confidence = 100 * probs[predicted_class_idx]
        
        st.metric(label="Predicted Posture", value=class_names[predicted_class_idx])
        if calibrate and model_choice == "CNN":
            st.caption(f"Confidence calibrated with temperature = {temperature:.1f}")
        
        if confidence >= 80:
            st.success(f"Confidence score: **{confidence:.2f}%** ✅")
        elif confidence >= 60:
            st.warning(f"Confidence score: **{confidence:.2f}%** ⚠️")
        else:
            st.error(f"Confidence score: **{confidence:.2f}%** ❌")
        
        # Top 3 predictions
        top3_idx = np.argsort(probs)[-3:][::-1]
        st.write("**Top 3 predictions:**")
        for idx in top3_idx:
            st.write(f"- {class_names[idx]}: {probs[idx]*100:.1f}%")
else:
    st.info(f"👈 No model found for '{model_choice}'. Click 'Train Model' first.")