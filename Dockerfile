FROM python

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

WORKDIR /app

RUN pip install matplotlib

RUN pip install seaborn

RUN pip install streamlit
