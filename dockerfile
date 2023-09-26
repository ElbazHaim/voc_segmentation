FROM pytorchlightning/pytorch_lightning:latest
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD [ "/bin/bash" ]
