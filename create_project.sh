#!/bin/bash

# Variables
PROJECT_NAME=$1
AUTHOR_NAME=$2
AUTHOR_EMAIL=$3
GITHUB_USERNAME=$4

# Check if all arguments are provided
if [ -z "$PROJECT_NAME" ] || [ -z "$AUTHOR_NAME" ] || [ -z "$AUTHOR_EMAIL" ] || [ -z "$GITHUB_USERNAME" ]; then
  echo "Usage: ./create_project.sh <project_name> <author_name> <author_email> <github_username>"
  exit 1
fi

# Template repository URL
TEMPLATE_REPO="https://github.com/m0574f4/python-package-template.git"

# Clone the template repository into the new project directory without git history
git clone --depth 1 "$TEMPLATE_REPO" "$PROJECT_NAME"

# Check if cloning was successful
if [ $? -ne 0 ]; then
  echo "Failed to clone the template repository."
  exit 1
fi

# Remove the .git directory to detach from the template repo
rm -rf "$PROJECT_NAME/.git"

# Navigate into the new project directory
cd "$PROJECT_NAME" || { echo "Failed to navigate to project directory"; exit 1; }

# Run the initialization script from the template
if [ -f ./init_project.sh ]; then
  ./init_project.sh "$PROJECT_NAME" "$AUTHOR_NAME" "$AUTHOR_EMAIL"
else
  echo "Initialization script not found. Skipping..."
fi

# Initialize a new git repository
git init

# Configure Git user information
git config user.email "$AUTHOR_EMAIL"
git config user.name "$AUTHOR_NAME"

# Add all files to the repository
git add .

# Commit the initial commit
git commit -m "Initial commit"

# Construct the remote repository URL using the GitHub username and project name
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$PROJECT_NAME.git"

# Add the remote repository
git remote add origin "$REMOTE_URL"

# Push to the remote repository
git branch -M main
git push -u origin main

# Output success message
echo "Project $PROJECT_NAME created and initialized successfully and pushed to $REMOTE_URL."
