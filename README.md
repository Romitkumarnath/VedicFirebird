# Vedic Astrology Bot

An AI-powered application that answers questions about Vedic astrology based on your PDF documents.

## Features

- Ask questions about Vedic astrology concepts
- Uses Claude AI for accurate, contextual responses
- Easy-to-use graphical interface
- Vector search for relevant document retrieval
- Document management capabilities

## Requirements

To use this application, you need:

1. An Anthropic API key (get one at https://console.anthropic.com/)
2. PDF documents about Vedic astrology (or use the default documents)

## Installation

### Option 1: Run from executable (Windows)

1. Download the latest release from the Releases section
2. Extract the ZIP file to a location of your choice
3. Run "Vedic Astrology Bot.exe"

### Option 2: Run from source code

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Anthropic API key (optional):
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   python run_astrology_app.py
   ```

## Using the Application

### First-time Setup

1. When you first run the application, go to the "Setup" tab
2. Enter your Anthropic API key
3. Optionally specify a different documents directory (leave empty to use the default)
4. Click "Initialize Bot"

### Asking Questions

1. Go to the "Chat" tab
2. Type your question in the input box
3. Click "Ask" or press Enter
4. The response will appear in the chat area

### Managing Documents

1. Go to the "Document Management" tab
2. Click "Show Document Status" to see loaded documents
3. Click "Reload All Documents" if you've added new files

## Building the Executable

If you want to build the executable yourself:

1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```
2. Run the build command:
   ```
   pyinstaller astrology_app.spec
   ```
3. The executable will be created in the `dist/Vedic Astrology Bot` directory

## Troubleshooting

- If you get an error about missing modules, try reinstalling the dependencies
- If the application crashes, check that your API key is valid
- For document loading issues, ensure your PDFs are valid and accessible

## License

This project is licensed under the MIT License - see the LICENSE file for details.