FROM python:3.11-slim

WORKDIR /app

# install dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m textblob.download_corpora

# copy the rest of the code
COPY . .

EXPOSE 8000
