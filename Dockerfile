FROM python:3.10-slim

WORKDIR /app

# Copy your code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Streamlit
EXPOSE 8501

# Required env for streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

# Run your streamlit app
CMD ["streamlit", "run", "src/your_main_file.py", "--server.address=0.0.0.0", "--server.port=8501"]
