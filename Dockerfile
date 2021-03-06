FROM python:3.7

RUN apt-get update && apt-get install -y python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app app/

RUN python app/app.py

EXPOSE 8080

CMD ["python", "app/app.py", "serve"]