i have this sh file, can you generate a windows bat file for me?


#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]
then
  # Create virtual environment
  echo "Creating virtual environment..."
  python3 -m venv venv

  source venv/bin/activate
  # Install required packages
  echo "Installing required packages..."
  pip install -r requirements.txt

fi

source venv/bin/activate


# Run Flask app
echo "Running Flask app..."
python app.py
