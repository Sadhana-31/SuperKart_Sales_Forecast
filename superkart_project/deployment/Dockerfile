FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Expose the Gradio port
EXPOSE 7860

# Launch the Gradio app
CMD ["python", "app.py"]
