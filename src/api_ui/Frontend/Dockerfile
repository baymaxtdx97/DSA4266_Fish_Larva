FROM python:3.8


WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get install libsm6 libxrender1 libfontconfig1 libice6 libgl1 -y

COPY ./ ./

EXPOSE 8051

ENTRYPOINT ["streamlit", "run", "frontend.py"]