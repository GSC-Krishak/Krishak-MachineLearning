FROM python:3.12

WORKDIR /app

ENV PORT 6969
ENV HOST 0.0.0.0

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 6969

CMD ["uvicorn", "crop_predictor:app", "--host", "0.0.0.0", "--port", "6969"]
