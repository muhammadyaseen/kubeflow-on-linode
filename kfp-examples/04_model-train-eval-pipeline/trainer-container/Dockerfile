FROM python:3.9

# Update packages
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

CMD ["echo", "Container started!"]
