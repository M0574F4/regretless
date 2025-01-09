#!/bin/bash

# Variables
PROJECT_NAME=$1
AUTHOR_NAME=$2
AUTHOR_EMAIL=$3

# Check if all arguments are provided
if [ -z "$PROJECT_NAME" ] || [ -z "$AUTHOR_NAME" ] || [ -z "$AUTHOR_EMAIL" ]; then
  echo "Usage: ./init_project.sh <project_name> <author_name> <author_email>"
  exit 1
fi

# Update setup.py with new project details
sed -i "s/your_project_name/$PROJECT_NAME/g" setup.py
sed -i "s/Your Name/$AUTHOR_NAME/g" setup.py
sed -i "s/youremail@example.com/$AUTHOR_EMAIL/g" setup.py
mkdir -p src/"$PROJECT_NAME"


# Set up virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pipenv
pipenv install --dev

echo "Project $PROJECT_NAME initialized successfully."
