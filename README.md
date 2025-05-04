# Personal LLM Mobile App

A mobile application that connects your Android device to local Large Language Models running on your PC.

## Project Overview

This project allows you to run powerful Large Language Models on your PC and interact with them through a simple mobile interface. The system consists of two main components:

1. **Python Server**: Runs on your PC and manages the LLM models
2. **Android App**: Provides a user-friendly interface to interact with the models

## Features

- **Multiple Model Support**: Load different models based on your needs
- **Model Management**: Load and unload models to manage memory usage
- **Streaming Responses**: Real-time streaming of LLM responses
- **System Prompts**: Customize model behavior with system instructions
- **Error Handling**: Robust error catching and reporting

## Project Components

### Server-side Components

- **server.py**: Flask server that handles API requests from the Android app
- **llm_manager.py**: Manages LLM operations, including loading models and generating responses
- **config.py**: Configuration file for model settings and parameters

### Android App Components

- **MainActivity.kt**: Main UI with model selection, load/unload controls, and prompt input
- **build.gradle.kts**: Dependencies and build configuration
- **AndroidManifest.xml**: App permissions and configuration

## Setup Instructions

### Server Setup

1. Install required Python packages:
   ```
   pip install flask flask-cors llama-cpp-python
   ```

2. Update the model paths in `config.py` to point to your local LLM models

3. Run the server:
   ```
   python server.py
   ```

4. Note the server IP address displayed in the console

### Android App Setup

1. Open the project in Android Studio

2. Update the `serverBaseUrl` in `MainActivity.kt` with your PC's IP address:
   ```kotlin
   private val serverBaseUrl = "http://YOUR_PC_IP:5000"
   ```

3. Build and run the app on your phone

4. Ensure your phone and PC are on the same network

## Usage

1. **Select a model** from the dropdown menu

2. **Load the model** by clicking "Load Model"

3. (Optional) Enter a **system prompt** to customize model behavior

4. Enter your prompt in the input field

5. Click **Send** to get a response from the model

6. When done, click **Unload Model** to free up memory

## API Endpoints

The server exposes the following API endpoints:

- **POST /query**: Send a prompt to the model and get a response
- **GET /models**: Get a list of available models
- **POST /model/load**: Load a specific model
- **POST /model/unload**: Unload the currently loaded model
- **GET /model/status**: Check the current model status

## Next Steps

- Add support for chat history
- Implement model parameter customization in the UI
- Add support for image generation models
- Improve UI with better formatting
- Add settings to configure server address
- Implement authentication for secure access

## Troubleshooting

- **Connection Issues**: Ensure your phone and PC are on the same network
- **Model Loading Errors**: Verify model paths in `config.py`
- **Slow Responses**: Adjust `max_tokens` parameter or use a smaller model
- **Timeout Errors**: Increase OkHttp timeout values in `MainActivity.kt`
