FROM python:3.12.2-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt 

EXPOSE 5000

CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "5000"]
