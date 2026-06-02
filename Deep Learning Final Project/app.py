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
Choose between **CNN** (custom) and **MobileNetV2** (transfer learning).  
Use the **Calibrate CNN** option to reduce overconfidence (60‑80% range).  
MobileNetV2 gives naturally higher confidence (80‑95%).
""")

# -------------------------------------------------------------
# 2. DATASET PATH
# -------------------------------------------------------------
DATASET_PATH = "./dataset"

# -------------------------------------------------------------
# 3. CUSTOM CNN TRAINING FUNCTION (128x128)
# -------------------------------------------------------------
def train_cnn(dataset_path):
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
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("cnn_model.keras", save_best_only=True, monitor='val_accuracy')
    ]

    history = model.fit(train_ds, epochs=25, validation_data=val_ds, callbacks=callbacks, verbose=1)
    return model, history, class_names

# -------------------------------------------------------------
# 4. MOBILENETV2 TRAINING FUNCTION (224x224)
# -------------------------------------------------------------
def train_mobilenet(dataset_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=16
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("mobilenet_model.keras", save_best_only=True, monitor='val_accuracy')
    ]

    history = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks, verbose=1)
    return model, history, class_names

# -------------------------------------------------------------
# 5. SIDEBAR MODEL SELECTION AND TRAINING
# -------------------------------------------------------------
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.radio(
    "Select Model Architecture",
    ("CNN", "MobileNetV2")
)

# Option to calibrate CNN confidence (temperature scaling)
if model_choice == "CNN":
    calibrate = st.sidebar.checkbox("Calibrate CNN confidence", value=True)
    temperature = st.sidebar.slider("Temperature (higher = lower confidence)", 1.0, 3.0, 1.5, 0.1) if calibrate else 1.0
else:
    calibrate = False
    temperature = 1.0

if st.sidebar.button("🚀 Train Model"):
    with st.spinner(f"Training {model_choice}... This may take a few minutes."):
        if model_choice == "CNN":
            model, history, class_names = train_cnn(DATASET_PATH)
            model.save("cnn_model.keras")
            st.session_state["history_cnn"] = history.history   # save for later
            st.success("✅ CNN model trained and saved as 'cnn_model.keras'")
        else:
            model, history, class_names = train_mobilenet(DATASET_PATH)
            model.save("mobilenet_model.keras")
            st.session_state["history_mobilenet"] = history.history
            st.success("✅ MobileNetV2 model trained and saved as 'mobilenet_model.keras'")
        
        if model is None:
            st.error("Training failed. Check dataset path.")
        else:
            # Display training plots immediately
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
            
            # Force refresh to load the newly saved model
            st.rerun()

# -------------------------------------------------------------
# 5b. SHOW TRAINING CHARTS (from saved history)
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
# 6. PREDICTION SECTION
# -------------------------------------------------------------
st.subheader("📸 Test Model Prediction")

# Determine model file based on selection
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
        
        # Preprocess image
        img = tf.keras.utils.load_img(uploaded_file, target_size=(expected_height, expected_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        
        with st.spinner("Classifying..."):
            logits = model.predict(img_array)  # these are already probabilities (softmax)
            if calibrate and model_choice == "CNN":
                # Apply temperature scaling to reduce overconfidence
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