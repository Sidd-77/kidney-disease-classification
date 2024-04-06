FROM python:3.12.2-slim-bullseye

RUN apt update
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD [ "streamlit", "run", "application.py" ]
EXPOSE 8501