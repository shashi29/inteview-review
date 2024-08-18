FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy the requirements file into the image
COPY ./requirements.txt /code/requirements.txt

# Update apt-get and install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code into the image
COPY ./app /code/app

# Define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
