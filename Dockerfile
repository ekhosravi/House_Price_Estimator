FROM python:3.9-slim

WORKDIR /app

COPY ./app .


RUN pip install --no-chache-dir -r requirements.txt

EXPOSE 8080

CMD ["python" , "app.py"]
