FROM python:3.7
EXPOSE 8501
WORKDIR .
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run main.py
