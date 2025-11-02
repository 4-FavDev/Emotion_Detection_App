from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode='categorical',
    batch_size=32
)

val_data = datagen.flow_from_directory(
    'dataset/val',
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode='categorical',
    batch_size=32
)

test_data = datagen.flow_from_directory(
    'dataset/test',
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode='categorical',
    batch_size=32
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40
)

model.save('emotion_model.h5')
print("✅ Model training complete!")

# Optional: evaluate on test set
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc}")
