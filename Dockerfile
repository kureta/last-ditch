FROM ovhcom/ai-training-pytorch

RUN apt update && apt install libsndfile1 -y
RUN git clone https://github.com/kureta/last-ditch.git
RUN cd last-ditch && pip install -r cloud.txt

CMD cd last-ditch && ./train.sh
