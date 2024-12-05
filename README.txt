# NFL Play Success Prediction Tool

This application is designed to help users predict the likelihood of success for NFL plays based on various input parameters. It uses a Random Forest model to analyze data and provide a success probability for the given scenario.

## Project Structure

The project contains the following files:

### HTML Files
- **`templates/index.html`**: The main input form where users provide play parameters like quarter, time, offense/defense teams, and play details.
- **`templates/analysis.html`**: Displays the success probability along with a visual representation of the football field and feedback based on the probability.

### Python Files
- **`app.py`**: The Flask web application, which handles routes, processes user input, and integrates with the machine learning model.
- **`rf_6242.joblib`**: The pre-trained Random Forest model used for predictions.
- **`scaler.joblib`**: The scaler used to normalize input features before feeding them into the model.

### CSS and Static Files
- **`static/style.css`**: The primary CSS file for styling the application.
- **Static Images/Assets**: Placeholder for any static assets like images if required in the future.

## Features

1. **User Input**: The user can input details like quarter, minutes, seconds, offense/defense teams, formation, and line of scrimmage details.
2. **Analysis Page**: Shows the success probability for the play and provides a feedback message based on the result.
3. **Field Visualization**: Renders a football field visualization on the analysis page using an HTML5 `<canvas>`.

## Prerequisites

To run this application locally, you need:

- Python 3.7 or higher
- Flask
- Joblib
- Numpy

You can install the required dependencies using the following command:

```bash
pip install flask numpy joblib
