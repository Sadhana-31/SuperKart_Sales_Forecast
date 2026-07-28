# Use a lightweight Python 3.9 base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all deployment files into the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose the Gradio port
EXPOSE 7860

# Launch the Gradio app
CMD ["python", "app.py"]
