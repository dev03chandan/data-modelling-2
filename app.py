from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
# Initialize the OpenAI client
# TODO: PLEASE INPUT OPENAI_API_KEY HERE
client = OpenAI(api_key=api_key)

import pandas as pd
import re
import numpy as np
import tensorflow as tf

# Imports for Preprocessing and Data Preparation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Imports for Model Training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import os

# Imports for Streamlit Application
import streamlit as st
from openai import OpenAI
import json
import tempfile
from zipfile import ZipFile


def load_and_explore_csv(file_path, text_column, label_column, num_rows_preview=5):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Validate required columns
        if text_column not in data.columns or label_column not in data.columns:
            return {
                "error": f"Columns '{text_column}' and '{label_column}' must exist in the dataset."
            }

        # General information
        num_rows = len(data)
        num_columns = len(data.columns)
        column_names = data.columns.tolist()

        # Null values
        null_values = data.isnull().sum().to_dict()
        null_percent = {
            col: f"{(val / num_rows) * 100:.2f}%" for col, val in null_values.items()
        }

        # Label distribution
        label_counts = data[label_column].value_counts().to_dict()
        total_labels = sum(label_counts.values())
        label_distribution = {
            k: f"{(v / total_labels) * 100:.2f}%" for k, v in label_counts.items()
        }

        # Text statistics
        text_lengths = data[text_column].dropna().apply(lambda x: len(str(x).split()))
        text_stats = {
            "average_length": text_lengths.mean(),
            "min_length": text_lengths.min(),
            "max_length": text_lengths.max(),
        }

        # Preview the dataset
        preview = (
            data[[text_column, label_column]]
            .head(num_rows_preview)
            .to_dict(orient="records")
        )

        return {
            "status": "success",
            "preview": preview,
            "general_info": {
                "num_rows": num_rows,
                "num_columns": num_columns,
                "column_names": column_names,
            },
            "null_values": {
                "counts": null_values,
                "percentages": null_percent,
            },
            "label_distribution": {
                "counts": label_counts,
                "percentages": label_distribution,
            },
            "text_statistics": text_stats,
        }
    except Exception as e:
        return {"error": str(e)}


def preprocess_text_data_with_options(
    file_path,
    text_column,
    label_column,
    handle_missing="drop",  # Options: 'drop', 'fill'
    fill_value="unknown",  # Used if handle_missing='fill'
    clean_text_options=None,  # Dict for cleaning options
    output_file="preprocessed_data.csv",
):
    """
    Preprocess text data with customizable options for missing values and text cleaning.
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Validate required columns
        if text_column not in data.columns or label_column not in data.columns:
            return {
                "error": f"Columns '{text_column}' and '{label_column}' must exist in the dataset."
            }

        # Handle missing values
        if handle_missing == "drop":
            data = data.dropna(subset=[text_column, label_column])
        elif handle_missing == "fill":
            data[text_column] = data[text_column].fillna(fill_value)
            data[label_column] = data[label_column].fillna(fill_value)
        else:
            return {"error": f"Invalid option for handle_missing: {handle_missing}"}

        # Default text cleaning options
        if clean_text_options is None:
            clean_text_options = {
                "lowercase": True,
                "remove_punctuation": True,
                "remove_numbers": True,
                "remove_extra_spaces": True,
            }

        # Text cleaning function
        def clean_text(text):
            if clean_text_options.get("lowercase", False):
                text = text.lower()
            if clean_text_options.get("remove_punctuation", False):
                text = re.sub(r"[^\w\s]", "", text)
            if clean_text_options.get("remove_numbers", False):
                text = re.sub(r"\d+", "", text)
            if clean_text_options.get("remove_extra_spaces", False):
                text = re.sub(r"\s+", " ", text)
            return text.strip()

        # Apply text cleaning
        data[text_column] = data[text_column].apply(clean_text)

        # Save the cleaned dataset
        data.to_csv(output_file, index=False)

        return {
            "status": "success",
            "message": "Data preprocessing complete.",
            "output_file": output_file,
            "num_rows": len(data),
            "cleaning_options_used": clean_text_options,
            "missing_value_handling": handle_missing,
        }
    except Exception as e:
        return {"error": str(e)}


def prepare_data_for_cnn(
    file_path,
    text_column,
    label_column,
    variable_name,
    num_words=10000,
    max_length=100,
    padding_type="post",
    truncating_type="post",
    test_size=0.2,
    validation_split=None,
    label_encoding="one-hot",
    oov_token="<OOV>",
):
    """
    Prepare data for training a CNN model by tokenizing text data and encoding labels.

    Parameters:
        file_path (str): Path to the dataset file (CSV).
        text_column (str): Name of the column containing text data.
        label_column (str): Name of the column containing labels.
        variable_name (str): Placeholder for additional variable usage.
        num_words (int): Maximum number of words to keep in the tokenizer vocabulary.
        max_length (int): Maximum length of sequences after padding.
        padding_type (str): Padding type ('post' or 'pre').
        truncating_type (str): Truncation type ('post' or 'pre').
        test_size (float): Proportion of data for the test set.
        validation_split (float): Proportion of data for the validation set (optional).
        label_encoding (str): Encoding type for labels ('one-hot' or 'integer').
        oov_token (str): Token to use for out-of-vocabulary words.

    Returns:
        dict: A dictionary containing tokenized data, word index, and train/test/validation splits.
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Tokenize text data
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data[text_column])
        sequences = tokenizer.texts_to_sequences(data[text_column])
        word_index = tokenizer.word_index

        # Pad sequences to ensure uniform input length
        padded_sequences = pad_sequences(
            sequences,
            maxlen=max_length,
            padding=padding_type,
            truncating=truncating_type,
        )

        # Encode labels based on the selected label encoding method
        if label_encoding == "one-hot":
            labels = pd.get_dummies(data[label_column]).values.astype(
                int
            )  # Ensure integer representation
        elif label_encoding == "integer":
            labels = (
                data[label_column].astype("category").cat.codes.values
            )  # Integer encoding of labels
        else:
            return {"error": f"Invalid label_encoding: {label_encoding}"}

        # Split data into train, test, and optionally validation sets
        result = {
            "status": "success",
            "tokenizer": tokenizer,
            "word_index": word_index,
            "label_encoding": label_encoding,
            "num_classes": (
                labels.shape[1] if label_encoding == "one-hot" else len(set(labels))
            ),
            "shapes": {
                "X_train": None,
                "X_val": None,
                "X_test": None,
                "y_train": None,
                "y_val": None,
                "y_test": None,
            },
        }

        if validation_split:
            # Perform initial split to isolate test set
            X_train, X_test, y_train, y_test = train_test_split(
                padded_sequences,
                labels,
                test_size=test_size + validation_split,
                random_state=42,
            )

            # Further split training data into training and validation sets
            val_size = validation_split / (test_size + validation_split)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42
            )

            result.update(
                {
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                }
            )

            result["shapes"].update(
                {
                    "X_train": X_train.shape,
                    "X_val": X_val.shape,
                    "X_test": X_test.shape,
                    "y_train": y_train.shape,
                    "y_val": y_val.shape,
                    "y_test": y_test.shape,
                }
            )
        else:
            # Perform a single split if no validation set is required
            X_train, X_test, y_train, y_test = train_test_split(
                padded_sequences, labels, test_size=test_size, random_state=42
            )

            result.update(
                {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }
            )

            result["shapes"].update(
                {
                    "X_train": X_train.shape,
                    "X_test": X_test.shape,
                    "y_train": y_train.shape,
                    "y_test": y_test.shape,
                }
            )

        return result

    except Exception as e:
        return {"error": str(e)}


def train_cnn_model_chatgpt(
    variable_name: str,
    embedding_dim: int = 100,
    model_layers: list = None,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    epochs: int = 10,
    batch_size: int = 32,
    validation_data: bool = True,
    early_stopping: bool = False,
    patience: int = 3,
    prepared_data: dict = {},
) -> dict:
    """
    Train a CNN model dynamically with user-defined architecture.

    Args:
        variable_name (str): Name of the variable containing preprocessed data.
        embedding_dim (int): Dimension of embedding layer.
        model_layers (list): List of dictionaries defining layer configurations.
        optimizer (str): Optimizer to use ('adam', 'sgd', 'rmsprop').
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_data (bool): Whether to use validation data during training.
        early_stopping (bool): Whether to enable early stopping.
        patience (int): Number of epochs to wait before early stopping.
        prepared_data (dict): Dictionary containing the prepared data from previous step.

    Returns:
        dict: A structured response with training results or errors.
    """
    try:
        # Validate prepared data
        if prepared_data is {}:
            return {
                "status": "error",
                "message": f"No prepared data provided for '{variable_name}'",
            }

        # Extract data and metadata
        X_train, y_train = prepared_data["X_train"], prepared_data["y_train"]
        X_val, y_val = prepared_data.get("X_val"), prepared_data.get("y_val")
        vocab_size = len(prepared_data["word_index"]) + 1
        label_encoding = prepared_data.get(
            "label_encoding", "integer"
        )  # Default to integer encoding
        num_classes = prepared_data.get("num_classes", None)

        # Ensure y_train is compatible with the model
        if label_encoding == "integer":
            # For integer-encoded labels, ensure correct shape
            if len(y_train.shape) == 1:
                y_train = np.expand_dims(y_train, axis=-1)
                if y_val is not None:
                    y_val = np.expand_dims(y_val, axis=-1)

        elif label_encoding == "one-hot":
            # For one-hot encoded labels, num_classes should match y_train.shape[1]
            if num_classes is None:
                num_classes = y_train.shape[1]
        else:
            return {
                "status": "error",
                "message": f"Unsupported label encoding: {label_encoding}",
            }

        # Build the model
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

        # Add layers based on model_layers configuration
        for layer in model_layers:
            if layer["type"] == "Conv1D":
                model.add(
                    Conv1D(
                        filters=layer.get("filters", 32),
                        kernel_size=layer.get("kernel_size", 3),
                        activation=layer.get("activation", "relu"),
                    )
                )
            elif layer["type"] == "MaxPooling1D":
                model.add(MaxPooling1D(pool_size=layer.get("pool_size", 2)))
            elif layer["type"] == "Flatten":
                model.add(Flatten())
            elif layer["type"] == "Dense":
                model.add(
                    Dense(
                        units=layer.get("units", 128),
                        activation=layer.get("activation", "relu"),
                    )
                )
            elif layer["type"] == "Dropout":
                model.add(Dropout(rate=layer.get("rate", 0.5)))

        # Add final output layer based on encoding type
        if label_encoding == "integer":
            # Integer-encoded labels require sparse_categorical_crossentropy
            model.add(Dense(num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
        elif label_encoding == "one-hot":
            # One-hot encoded labels require categorical_crossentropy
            model.add(Dense(num_classes, activation="softmax"))
            loss = "categorical_crossentropy"

        # Configure optimizer
        if optimizer.lower() == "adam":
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            opt = SGD(learning_rate=learning_rate)
        elif optimizer.lower() == "rmsprop":
            opt = RMSprop(learning_rate=learning_rate)
        else:
            return {"status": "error", "message": f"Unsupported optimizer: {optimizer}"}

        # Compile model
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy"],
        )

        # Configure callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss" if validation_data else "loss",
                    patience=patience,
                    restore_best_weights=True,
                )
            )

        # Train the model
        if validation_data and X_val is not None and y_val is not None:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
            )
        else:
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
            )

        # Convert history to regular Python types for JSON serialization
        history_dict = {
            key: [float(v) for v in value] for key, value in history.history.items()
        }

        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        model_path = os.path.join(models_dir, f"{variable_name}_model.keras")
        model.save(model_path)

        return {
            "status": "success",
            "message": "Model training complete",
            "model_variable_name": f"{variable_name}_model",
            "history": history_dict,
            "model": model,
            "model_path": model_path,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


system_prompt = """You are an AI assistant specialized in guiding users through building text classification models using CNNs, with expertise akin to a skilled Data Scientist Guide. Your assistance spans four main stages of the project lifecycle:

1. **Data Input and Exploration**:
   - Guide users in loading datasets (CSV format) and verifying essential columns.
   - Help explore the dataset by providing summaries, label distributions, missing values, and basic statistics on text data.
   - Preview the dataset to identify any immediate anomalies or opportunities for improvement.
   - For this step only ask the user to upload the dataset in the sidebar of the page.

2. **Data Preprocessing**:
   - Assist in cleaning and transforming the dataset.
   - Offer user-customizable options for handling missing values,remove extra spaces, punctuation removal.
   - Ensure explanations are provided for preprocessing decisions, and respect user preferences when automating steps.
   - Use function calling to get this step done.

3. **Data Preparation for CNNs**:
   - Guide users in tokenizing and padding text, creating train-validation-test splits, and encoding labels for CNN compatibility.
   - Provide flexibility for user input on parameters like maximum sequence length, vocabulary size, and test size.
   - Ensure the processed data is prepared in a form suitable for CNNs, with detailed feedback on the steps.

4. **Model Training**:
   - Help design CNN models, allowing users to customize the architecture (e.g., number of layers, filter sizes, and activation functions).
   - Support optimizer selection, hyperparameter tuning, and training configurations like batch size, epochs, and callbacks.
   - Encourage iterative improvement by suggesting adjustments based on training outcomes.

### Communication Guidelines:
- Adjust your explanations and language to match the user's expertise based on their responses. Use simplified terms for beginners and appropriate jargon for advanced users.
- Avoid overwhelming users with too many questions at once. Break down interactions into logical sequences and offer explanations step-by-step.
- Promptly clarify user inputs and ask follow-up questions one at a time to ensure mutual understanding.
- Emphasize best practices, explaining why a particular method or parameter setting is recommended.
- Encourage users to make informed decisions while providing default values or recommendations as needed.

### Interaction Flow:
- During **model training**, split input collection across multiple prompts to gather detailed information on each aspect (e.g., model structure, optimizer, and training hyperparameters).
- Be proactive but patient, ensuring the user is comfortable and well-informed before proceeding to the next stage.

Your primary goal is to ensure a smooth and effective experience for the user while building their CNN-based text classification project."""


tools = [
    {
        "type": "function",
        "function": {
            "name": "load_and_explore_csv",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "description": "Loads a CSV file and performs exploratory data analysis",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preprocess_text_data_with_options",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": [
                    "file_path",
                    "text_column",
                    "label_column",
                    "handle_missing",
                    "fill_value",
                    "clean_text_options",
                    "output_file",
                ],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file containing the text data",
                    },
                    "fill_value": {
                        "type": "string",
                        "description": "Value to fill for missing data if handle_missing is set to 'fill'",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the preprocessed data as a CSV file",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Name of the column containing text data",
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Name of the column containing labels associated with the text data",
                    },
                    "handle_missing": {
                        "enum": ["drop", "fill"],
                        "type": "string",
                        "description": "Method to handle missing values; options are 'drop' or 'fill'",
                    },
                    "clean_text_options": {
                        "type": "object",
                        "required": [
                            "lowercase",
                            "remove_punctuation",
                            "remove_numbers",
                            "remove_extra_spaces",
                        ],
                        "properties": {
                            "lowercase": {
                                "type": "boolean",
                                "description": "Whether to convert text to lowercase",
                            },
                            "remove_numbers": {
                                "type": "boolean",
                                "description": "Whether to remove numbers from text",
                            },
                            "remove_punctuation": {
                                "type": "boolean",
                                "description": "Whether to remove punctuation from text",
                            },
                            "remove_extra_spaces": {
                                "type": "boolean",
                                "description": "Whether to remove extra spaces from text",
                            },
                        },
                        "description": "Dictionary containing options for text cleaning",
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
            "description": "Preprocess text data with customizable options for missing values and text cleaning.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_data_for_cnn",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": [
                    "file_path",
                    "text_column",
                    "label_column",
                    "variable_name",
                    "num_words",
                    "max_length",
                    "padding_type",
                    "truncating_type",
                    "test_size",
                    "validation_split",
                    "label_encoding",
                    "oov_token",
                ],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file containing the dataset.",
                    },
                    "num_words": {
                        "type": "number",
                        "description": "Maximum number of words to keep in the tokenizer. Default is 10000.",
                    },
                    "oov_token": {
                        "type": "string",
                        "description": "Token used to represent out-of-vocabulary words. Default is '<OOV>'.",
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Proportion of the dataset to include in the test split. Default is 0.2.",
                    },
                    "max_length": {
                        "type": "number",
                        "description": "Maximum length of sequences after padding. Default is 100.",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Name of the column containing the text data.",
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Name of the column containing the labels.",
                    },
                    "padding_type": {
                        "type": "string",
                        "description": "Type of padding to apply: 'pre' or 'post'. Default is 'post'.",
                    },
                    "variable_name": {
                        "type": "string",
                        "description": "Name under which prepared data should be stored.",
                    },
                    "label_encoding": {
                        "type": "string",
                        "description": "Method for encoding labels. Options are 'one-hot' or 'integer'. Default is 'one-hot'.",
                    },
                    "truncating_type": {
                        "type": "string",
                        "description": "Type of truncation to apply: 'pre' or 'post'. Default is 'post'.",
                    },
                    "validation_split": {
                        "type": "number",
                        "description": "Proportion of training data to use as validation data. Default is None.",
                    },
                },
                "additionalProperties": False,
            },
            "description": "Prepare text data for CNN training with customizable options and store it with a variable name.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_cnn_model_chatgpt",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": [
                    "variable_name",
                    "embedding_dim",
                    "model_layers",
                    "optimizer",
                    "learning_rate",
                    "epochs",
                    "batch_size",
                    "validation_data",
                    "early_stopping",
                    "patience",
                ],
                "properties": {
                    "epochs": {
                        "type": "number",
                        "description": "Number of epochs to train the model.",
                    },
                    "patience": {
                        "type": "number",
                        "description": "Number of epochs with no improvement after which training will be stopped.",
                    },
                    "optimizer": {
                        "type": "string",
                        "description": "The optimizer to use for training the model.",
                    },
                    "batch_size": {
                        "type": "number",
                        "description": "Number of samples per gradient update.",
                    },
                    "model_layers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "type",
                                "filters",
                                "kernel_size",
                                "activation",
                                "pool_size",
                                "units",
                                "rate",
                            ],
                            "properties": {
                                "rate": {
                                    "type": "number",
                                    "description": "Dropout rate (for Dropout layer).",
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Type of layer (e.g., Conv1D, MaxPooling1D, Flatten, Dense, Dropout).",
                                },
                                "units": {
                                    "type": "number",
                                    "description": "Number of units (for Dense layer).",
                                },
                                "filters": {
                                    "type": "number",
                                    "description": "Number of filters (for Conv1D layer).",
                                },
                                "pool_size": {
                                    "type": "number",
                                    "description": "Size of the pooling window (for MaxPooling1D layer).",
                                },
                                "activation": {
                                    "type": "string",
                                    "description": "Activation function (e.g., 'relu', 'sigmoid', etc.).",
                                },
                                "kernel_size": {
                                    "type": "number",
                                    "description": "Size of the kernel (for Conv1D layer).",
                                },
                            },
                            "additionalProperties": False,
                        },
                        "description": "List defining the layers of the CNN model.",
                    },
                    "embedding_dim": {
                        "type": "number",
                        "description": "The dimensionality of the embedding layer.",
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate for the optimizer.",
                    },
                    "variable_name": {
                        "type": "string",
                        "description": "The name of the variable holding the training data.",
                    },
                    "early_stopping": {
                        "type": "boolean",
                        "description": "Indicates whether to use early stopping during training.",
                    },
                    "validation_data": {
                        "type": "boolean",
                        "description": "Indicates whether to use validation data during training.",
                    },
                },
                "additionalProperties": False,
            },
            "description": "Train a CNN model dynamically with user-defined architecture for OpenAI function calling.",
        },
    },
]


# STREAMLIT APP STARTS HERE


# Function to generate response from the OpenAI API
def get_openai_response(messages):
    completion = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools
    )
    return (
        completion.choices[0].message.content,
        completion.choices[0].message.tool_calls,
    )


# Streamlit UI
st.title("Data Modelling Chatbot 🤖")

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# Add this near the top of the file, after imports
if not os.path.exists("processed_data"):
    os.makedirs("processed_data")

if "saved_models" not in st.session_state:
    st.session_state.saved_models = {}

# Add this near the top where other session state variables are initialized
if "prepared_data" not in st.session_state:
    st.session_state.prepared_data = {}

# Initialize chat history with a system prompt
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hey there! How can I help you today?"},
    ]

# Display chat messages, but exclude the system message
for message in st.session_state.messages:
    if message["role"] in [
        "user",
        "assistant",
    ]:  # Skip the system message or function call
        with st.chat_message(message["role"]):
            st.write(message["content"])


# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hey there! How can I help you today?"},
    ]


# Add this in the sidebar, before the clear chat button
with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = temp_file_path

        # Preview the uploaded data
        df = pd.read_csv(temp_file_path)
        st.write("Available columns:")
        st.write(df.columns.tolist())

        # Column selectors
        st.session_state.text_column = st.selectbox("Select text column", df.columns)
        st.session_state.label_column = st.selectbox("Select label column", df.columns)

    st.button("Clear Chat History", on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    function_response = None
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the entire message history to OpenAI for context
                response, tool_calls = get_openai_response(st.session_state.messages)

                # Process function call if any
                if tool_calls:
                    # Simulate calling the function
                    tool_call = tool_calls[0]
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # For the sake of this test, we'll handle the function call
                    if function_name == "load_and_explore_csv":
                        if "uploaded_file_path" in st.session_state:
                            arguments["file_path"] = st.session_state.uploaded_file_path
                            arguments["text_column"] = st.session_state.text_column
                            arguments["label_column"] = st.session_state.label_column
                            function_response = load_and_explore_csv(**arguments)
                        else:
                            function_response = {
                                "error": "Please upload a dataset first."
                            }

                    elif function_name == "preprocess_text_data_with_options":
                        if "uploaded_file_path" in st.session_state:
                            arguments["file_path"] = st.session_state.uploaded_file_path
                            arguments["text_column"] = st.session_state.text_column
                            arguments["label_column"] = st.session_state.label_column
                            # Create a temporary output file path
                            output_file = os.path.join(
                                "processed_data",
                                f"preprocessed_{os.path.basename(st.session_state.uploaded_file_path)}",
                            )
                            arguments["output_file"] = output_file
                            function_response = preprocess_text_data_with_options(
                                **arguments
                            )
                            function_response_str = json.dumps(
                                function_response, default=str
                            )
                        else:
                            function_response = {
                                "error": "Please upload a dataset first."
                            }

                    elif function_name == "prepare_data_for_cnn":
                        if "uploaded_file_path" in st.session_state:
                            arguments["file_path"] = st.session_state.uploaded_file_path
                            arguments["text_column"] = st.session_state.text_column
                            arguments["label_column"] = st.session_state.label_column

                            # Create a variable name if not provided
                            if "variable_name" not in arguments:
                                arguments["variable_name"] = "default_dataset"

                            # Call the function and store result directly in session state
                            try:
                                prepared_data = prepare_data_for_cnn(**arguments)
                                if (
                                    "status" in prepared_data
                                    and prepared_data["status"] == "success"
                                ):
                                    st.session_state.prepared_data[
                                        arguments["variable_name"]
                                    ] = {
                                        "X_train": prepared_data["X_train"],
                                        "X_test": prepared_data["X_test"],
                                        "y_train": prepared_data["y_train"],
                                        "y_test": prepared_data["y_test"],
                                        "tokenizer": prepared_data["tokenizer"],
                                        "word_index": prepared_data["word_index"],
                                        "shapes": prepared_data["shapes"],
                                        "num_classes": prepared_data["num_classes"],
                                        "label_encoding": prepared_data[
                                            "label_encoding"
                                        ],
                                    }
                                    # If validation split was used
                                    if "X_val" in prepared_data:
                                        st.session_state.prepared_data[
                                            arguments["variable_name"]
                                        ].update(
                                            {
                                                "X_val": prepared_data["X_val"],
                                                "y_val": prepared_data["y_val"],
                                            }
                                        )
                                    function_response = {
                                        "status": "success",
                                        "variable_name": arguments["variable_name"],
                                    }
                                else:
                                    function_response = (
                                        prepared_data  # Contains error message
                                    )
                            except Exception as e:
                                function_response = {"error": str(e)}

                            function_response_str = json.dumps(
                                function_response, default=str
                            )
                        else:
                            function_response = {
                                "error": "Please upload a dataset first."
                            }

                    elif function_name == "train_cnn_model_chatgpt":
                        if "prepared_data" not in st.session_state:
                            function_response = {
                                "error": "No prepared data available. Please prepare data first."
                            }
                        else:
                            variable_name = arguments.get(
                                "variable_name", "default_dataset"
                            )
                            if variable_name not in st.session_state.prepared_data:
                                function_response = {
                                    "error": f"No prepared data found for variable name: {variable_name}"
                                }
                            else:
                                try:
                                    # Get the prepared data from session state
                                    prepared_data = st.session_state.prepared_data[
                                        variable_name
                                    ]

                                    # Add prepared_data to arguments
                                    arguments["prepared_data"] = prepared_data

                                    # Call the training function
                                    result = train_cnn_model_chatgpt(**arguments)

                                    # Store the trained model if training was successful
                                    if result["status"] == "success":
                                        if "models" not in st.session_state:
                                            st.session_state.models = {}
                                        st.session_state.models[
                                            result["model_variable_name"]
                                        ] = result["model"]
                                        st.session_state.saved_models[
                                            result["model_variable_name"]
                                        ] = result["model_path"]
                                        model_path = result["model_path"]

                                        zip_path = (
                                            f"{os.path.splitext(model_path)[0]}.zip"
                                        )

                                        with ZipFile(zip_path, "w") as zipf:
                                            zipf.write(
                                                model_path, os.path.basename(model_path)
                                            )

                                        result["download_path"] = zip_path

                                        del result["model"]

                                    function_response = result
                                except Exception as e:
                                    function_response = {"error": str(e)}

                    function_response_str = json.dumps(function_response, default=str)

                    # Add the assistant's message with tool_calls first
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )

                    # Then add the tool response
                    st.session_state.messages.append(
                        {
                            "role": "tool",
                            "content": function_response_str,
                            "tool_call_id": tool_call.id,
                        }
                    )

                    # Make the API call
                    completion = client.chat.completions.create(
                        model="gpt-4o", messages=st.session_state.messages
                    )

                    response = completion.choices[0].message.content

                st.write(response)

                if function_response:
                    # Add this after processing the assistant's response
                    if "download_path" in function_response:
                        with open(function_response["download_path"], "rb") as f:
                            st.download_button(
                                label="Download Trained Model",
                                data=f,
                                file_name=f"{variable_name}_model.zip",
                                mime="application/zip",
                            )

        # Add assistant response to chat history
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
