import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split



##############################################   

def prepare_dataset(dataset_dir, class_labels, target_size=(128, 128)):
    X = []
    y = []

    for label_index, class_label in enumerate(class_labels):
        label_folder = os.path.join(dataset_dir, class_label)
        image_files = os.listdir(label_folder)

        for img_file in image_files:
            img_path = os.path.join(label_folder, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.vgg16.preprocess_input(img)
            X.append(img)
            y.append(label_index)

    X = np.array(X)
    y = np.array(y)

    return X, y

#################################

def vgg_model(input_shape, weights, num_classes=9):
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 2 output classes
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

##############################################   

def resnet_model(input_shape, weights, num_classes=9):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 2 output classes
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

##############################################   

def cv_train_and_evaluate_model(dataset_dir, test_dataset_dir, class_labels, model_fn, weights, input_shape, title, file_name, cv, epochs, batch_size):
    
    print('Class labels: ', class_labels)

    # Prepare dataset with resized images
    X, y = prepare_dataset(dataset_dir, class_labels, target_size=input_shape[:2])
    
    # Train the model
    model_history, trained_model = cv_train_model_v2(model_fn, X, y, cv, epochs, batch_size, input_shape, weights)
    
    # Plot training history
    plot_train_history(model_history, title, file_name, cv)
    
    
    # Test on data
    test_metrics, confusion_matrix = test_on_data(test_dataset_dir, trained_model, class_labels)
    
    # Print test metrics
    print(test_metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, class_labels, f'{title} on Test Data')

    return trained_model, test_metrics, confusion_matrix


##############################################   

def plot_train_history(fold_metrics_df, title, file_name, cv):

    if cv == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns of subplots for cv=1
        axes = np.array(axes).reshape(1, 2)  # Ensure axes are always 2D for consistent indexing
    else:
        fig, axes = plt.subplots(cv, 2, figsize=(20, 5 * cv))  # cv rows, 2 columns of subplots

    y_ticks = np.arange(0, 1.1, 0.1)

    # Plot training loss vs validation loss and training accuracy vs validation accuracy for each fold
    for i in range(cv):  # Iterate over each fold
        axes[i, 0].plot(fold_metrics_df['Train Loss'][i], label='Train Loss')
        axes[i, 0].plot(fold_metrics_df['Val Loss'][i], label='Val Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()

        axes[i, 1].plot(fold_metrics_df['Train Accuracy'][i], label='Train Accuracy')
        axes[i, 1].plot(fold_metrics_df['Val Accuracy'][i], label='Val Accuracy')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Accuracy')
#         axes[i, 1].set_ylim(0, 1.1)
        axes[i, 1].set_yticks(y_ticks)
        axes[i, 1].legend()

    # Set general title for the entire figure
    fig.suptitle(title, fontsize=16)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


##############################################   
def evaluate_test_set(model, X_test, y_test):
    """
    Evaluate the trained model on a test set and return performance metrics as a dictionary with two-digit precision.

    Args:
    - model: Trained VGG model.
    - X_test (numpy.ndarray): Test set features.
    - y_test (numpy.ndarray): Test set labels.
    
    Returns:
    - metrics_dict (dict): Dictionary containing performance metrics.
    """
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    test_loss = round(test_loss, 4)
    test_accuracy = round(test_accuracy, 4)

    # Get model predictions on test data
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Calculate additional performance metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    sensitivity = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    
    # Round metrics to two-digit precision
    f1 = round(f1, 2)
    sensitivity = round(sensitivity, 2)
    precision = round(precision, 2)

    # Calculate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    tn = conf_matrix[1, 1]  # True negatives for "Without Structural Pathology"
    fp = conf_matrix[1, 0]  # False positives for "Without Structural Pathology"
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Create dictionary to store metrics
    metrics_dict = {
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'F1 Score': f1,
        'Sensitivity (Recall)': sensitivity,
        'Precision': precision,
        'Specificity (With Structural Pathology)': specificity,
    }

    return metrics_dict, conf_matrix

##############################################   

def test_on_data(dataset_dir, model, class_labels):

    X_test, y_test = prepare_dataset(dataset_dir, class_labels, target_size=(128, 128))
    metrics_dict, cf_matrix = evaluate_test_set(model, X_test, y_test)

    return metrics_dict, cf_matrix

##############################################  
  
def plot_confusion_matrix(conf_matrix, class_names, title):
    """
    Plot the confusion matrix as a heatmap.

    Args:
    - conf_matrix (numpy.ndarray): Confusion matrix array.
    - class_names (list): List of class names (labels).
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    
   
##############################################    
#     return metrics, model
def cv_train_model_v2(model_fn, X, y, cv, epochs, batch_size, input_shape, weights):
    # Create and compile the model with the given input shape
    model = model_fn(input_shape, weights) #model = model_fn(input_shape=(128, 128, 3))

    # Split the data into training and validation sets manually (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy}")

    # Round the values before storing in metrics
    rounded_train_loss = [round(loss, 2) for loss in history.history['loss']]
    rounded_train_accuracy = [round(acc, 2) for acc in history.history['accuracy']]
    rounded_val_loss = [round(loss, 2) for loss in history.history['val_loss']]
    rounded_val_accuracy = [round(acc, 2) for acc in history.history['val_accuracy']]
    rounded_val_accuracy_mean = round(val_accuracy, 2)

    # Store the metrics in a dictionary
    metrics = {
        'Train Loss': rounded_train_loss,
        'Train Accuracy': rounded_train_accuracy,
        'Val Loss': rounded_val_loss,
        'Val Accuracy': rounded_val_accuracy,
        'Val Accuracy Mean': rounded_val_accuracy_mean
    }

    metrics_df = pd.DataFrame([metrics])

    return metrics_df, model

##############################################   

def train_and_evaluate_model(dataset_dir, test_dataset_dir, class_labels, model, input_shape, title, file_name):
    print('Class labels: ', class_labels)

    # Prepare dataset with resized images
    X, y = prepare_dataset(dataset_dir, class_labels, target_size=input_shape[:2])
    
    # Train the model
    model_history, trained_model = train_model(model, X, y, title=title, file_name=file_name)
    
    # Test on data
    test_metrics, confusion_matrix = test_on_data(test_dataset_dir, trained_model, class_labels)
    
    # Print test metrics
    print(test_metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_matrix, class_labels, f'{title} on Test Data')

    return trained_model, test_metrics, confusion_matrix

    
