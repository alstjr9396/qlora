FROM python:3.10.13-slim

WORKDIR /mnt/storage1/fineTuning-qlora
COPY . .
ENV PYTHONPATH /app
ENV HF_HOME=/mnt/storage1/fineTuning-qlora

RUN pip install pip==23.3.1 && \
	pip install -r requirements.txt
    
ENTRYPOINT [ "python3", "tuning.py" ]