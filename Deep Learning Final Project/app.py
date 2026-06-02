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
This app uses **MobileNetV2** to classify 5 types of body postures with high accuracy.  
*One arm crossed, Arms Crossed High, Arms Crossed Low, Arms Open/Neutral, and Hands in Pockets*.
""")

# -------------------------------------------------------------
# 2. DATASET PATH
# -------------------------------------------------------------
DATASET_PATH = "./dataset"

# -------------------------------------------------------------
# 3. TRAINING WITH TRANSFER LEARNING (MobileNetV2)
# -------------------------------------------------------------
def train_posture_model(dataset_path):
    if not os.path.exists(dataset_path):
        return None, None, "Dataset path not found. Please ensure 'dataset' folder is in the same directory as app.py"

    # Load datasets – use 224x224 for better transfer learning performance
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),   # MobileNetV2 input size
        batch_size=16            # adjust based on your GPU/CPU memory
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
    st.write(f"📂 Detected classes: {class_names}")

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data augmentation (to help generalization)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.1),
    ])

    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False   # Freeze the base model

    # Build the full model
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),   # MobileNetV2 expects [0,1] input
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # Compile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks: early stopping + save best model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("posture_model.keras", save_best_only=True, monitor='val_accuracy')
    ]

    # Train for more epochs (transfer learning converges fast)
    history = model.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, class_names

# -------------------------------------------------------------
# 4. SIDEBAR TRAINING BUTTON
# -------------------------------------------------------------
if st.sidebar.button("🚀 Train Model"):
    with st.spinner("Training with MobileNetV2... This may take a few minutes."):
        model, history, class_names = train_posture_model(DATASET_PATH)
        if model is None:
            st.error(history)
        else:
            st.success("✅ Model trained successfully! Best model saved as 'posture_model.keras'")
            
            # Plot accuracy and loss
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
            st.info("💡 Tip: For even higher confidence, train for more epochs or unfreeze some layers of the base model.")

# -------------------------------------------------------------
# 5. UPLOAD & PREDICT UI
# -------------------------------------------------------------
st.subheader("📸 Test Model Prediction")

if os.path.exists("posture_model.keras"):
    trained_model = tf.keras.models.load_model("posture_model.keras")
    
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
        
        # Preprocess exactly as training: resize to 224x224
        img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        # No manual rescaling – the model includes Rescaling(1./255)
        
        with st.spinner("Classifying..."):
            predictions = trained_model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = 100 * predictions[0][predicted_class_idx]
        
        st.metric(label="Predicted Posture", value=class_names[predicted_class_idx])
        st.success(f"Confidence score: **{confidence:.2f}%**")
        
        # Show top 3 predictions
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        st.write("**Top 3 predictions:**")
        for idx in top3_idx:
            st.write(f"- {class_names[idx]}: {predictions[0][idx]*100:.1f}%")
else:
    st.info("👈 Click 'Train Model' in the sidebar to train the high-accuracy transfer learning model.")