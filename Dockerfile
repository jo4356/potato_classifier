# This Dockerfile is from https://github.com/thedirtyfew/dash-docker-mwe

FROM python:3.10

# Create a working directory.
RUN mkdir wd
WORKDIR wd

# Install Python dependencies.
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the codebase into the image
COPY model/ ./model
COPY src/ ./src


# Finally, run gunicorn.
CMD [ "gunicorn", "--workers=1", "--threads=1", "-b 0.0.0.0:8000", "app:server"]