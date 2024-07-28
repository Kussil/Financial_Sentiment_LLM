# Use the official lightweight Python image.
FROM python:3.11

# Set the working directory in the container.
WORKDIR /app

# Add necessary GPG keys and install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg2 && \
    apt-get install -y --no-install-recommends dirmngr && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 04EE7237B7D453EC && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 648ACFD622F3D138 && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app_demo.py when the container launches
CMD ["streamlit", "run", "app_demo.py"]
