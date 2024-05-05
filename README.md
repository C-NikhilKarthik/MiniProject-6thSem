# Transformative Attendance System

Under the guidance of Dr. Vivekraj V K

## Objective

Traditional attendance tracking poses challenges such as time-consumption, errors, and inefficiency in large classes. Our innovative system seamlessly integrates cutting-edge computer vision models to address these issues.

## Scope

The system features:

- User-friendly app for faculty
- Real-time face detection and recognition
- Automated excel sheet population
- Detailed attendance analysis using graphs

## Challenges

1. Varying image quality based on faculty's mobile devices
2. Varied processing speeds on different devices
3. Model performance dependent on dataset diversity
   - Challenge in obtaining a diverse dataset due to privacy concerns

## Target Audience

Intended for schools and colleges, our system streamlines attendance tracking. A mobile app will be developed for faculty to capture, monitor, and analyze student attendance.

## Technology Stack

- React-Native / Flutter
- Google Sheets API
- Tensorflow
- Keras
- OpenCV

## Project Timeline (Gantt Chart)

![Gantt Chart](link-to-image)

- **Introduction:** Overview of the project timeline.
- **Overview of Tasks:** Key tasks from research to application development.
- **Representation of Weeks:** Timeline and task structure.
- **Tasks Spanning Multiple Weeks:** Factors influencing extended durations.
- **Task Details:** Task names, dates, responsible team members, and status.
- **Adjustments and Updates:** Flexibility to accommodate changes.
- **Visual Aid in Understanding:** Importance in project management.
- **Conclusion:** Significance of the Gantt chart in effective planning.

## Team Members

- Anand Kumar Singh (21BCS009)
- Avaneesh Sundararajan (21BCS020)
- C Nikhil Karthik (21BCS024)
- Karthik Avinash (21BCS052)

## Appendix: Installation Guideline for App
Following are the steps to install and run the app locally on your PC using Expo.

### Install from GitHub

1. **Clone the Repository:**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to clone the repository.
   - Run the following command to clone the repository:
     ```
     git clone https://github.com/C-NikhilKarthik/MiniProject-6thSem
     ```

2. **Set Up Conda Environment:**
   - Make sure you have Conda installed on your system. If not, you can download and install Miniconda from [Conda website](https://docs.conda.io/en/latest/miniconda.html).
   - Open your terminal or command prompt.
   - Navigate to the directory where you cloned the repository.
   - Run the following command to create a new Conda environment with Python version 3.10.11:
     ```
     conda create --name myenv python=3.10.11
     ```
   - Activate the Conda environment by running:
     ```
     conda activate myenv
     ```
5. **Install Dependencies for Expo (Client):**
   - Navigate to the root directory of the cloned repository.
   - Run the following command to install the required npm packages:
     ```
     npm install
     ```
   - This will install all the necessary dependencies for the Expo app.

6. **Install Dependencies for Flask Server (Server):**
   - Navigate to the `server` folder within the cloned repository.
   - Run the following command to install the required Python packages using pip:
     ```
     pip install -r requirements.txt
     ```
   - This will install all the necessary dependencies for the Flask server.

7. **Download Pre-Trained Models (Optional):**
   - If the app utilizes pre-trained machine learning models, you'll need to download them from the provided Google Drive link.
   - Once downloaded, create a folder named `models-pkl` inside the `server` directory of the cloned repository.
   - Place the downloaded model files into the `models-pkl` folder.

8. **Run the Servers:**
   - First, start the Flask server:
     - Navigate to the `server` folder.
     - Run the following command to start the server:
       ```
       python app.py
       ```
   - Next, start the Expo development server:
     - Run the following command to start the development server for the Expo app:
       ```
       npm start -- --reset-cache
       ```
     - This command will open the Expo developer tools in your default web browser.
     - Use the Expo Go app on your mobile device to scan the QR code displayed in the Expo developer tools.

9. **Accessing the App:**
   - After both servers have started successfully, you can access the app locally on your mobile device using the Expo Go app.

Now, you have successfully installed and run the app locally on your PC using Expo. You can start developing and testing the app according to your requirements.
