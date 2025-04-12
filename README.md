# Personal LLM Mobile App

A simple Android app that connects to a local LLM running on my PC.

## Project Components

### Android App
- Simple Jetpack Compose UI with a text input field and response display
- Makes HTTP requests to a local server running on my PC
- Displays responses from the LLM

### Python Server
- Flask server that receives requests from the Android app
- Processes requests and forwards them to the local LLM
- Returns responses back to the app

## Setup Instructions

### Server Setup
1. Install required Python packages:
   ```
   pip install flask flask-cors
   ```
2. Place server.py in your preferred directory
3. Run the server:
   ```
   python server.py
   ```
4. Note the server IP address displayed in the console

### Android App Setup
1. Update the `serverUrl` in MainActivity.kt with your PC's IP address
2. Build and run the app on your phone
3. Ensure your phone and PC are on the same network

## Usage
1. Enter a prompt in the text field
2. Tap "Send"
3. View the response from your LLM

## Next Steps
- Implement actual LLM integration on the server side
- Add message history
- Improve UI with better formatting
- Add settings to configure server address
