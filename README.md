# React-Flask Application

This repository contains an AI assistant application using React for the frontend and Python Flask for the backend.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- [Node.js](https://nodejs.org/) (which includes npm)
- [Python](https://www.python.org/downloads/) (version 3.7 or later)
- [pip](https://pip.pypa.io/en/stable/installing/)

## Getting Started

### Setting up the Backend

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask server:
   ```bash
   python app.py
   ```
   The Flask server will start running on `http://localhost:5000`.

### Setting up the Frontend

1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd ../frontend
   ```
2. Install the necessary npm packages:
   ```bash
   npm install
   ```
3. Start the React application:
   ```bash
   npm start
   ```
   This will run the React app and typically opens `http://localhost:3000` in your default web browser. The React app will connect to the Flask backend for any API requests.

## Exploring the App

With both servers running, you can develop and test the full stack application. The React frontend will send requests to the Flask backend, and you can modify either part of the stack as needed for your project.

Feel free to explore and expand this basic setup to suit your project needs!

---

This GitHub description outlines a clear path for setting up and running a basic full-stack application using React and Flask, providing a straightforward guide for new users.
