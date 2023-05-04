FROM datamachines/cudnn_tensorflow_opencv:11.6.2_2.9.1_4.6.0-20220815

WORKDIR /home/jupyter

RUN python3 -m pip install --no-cache --upgrade setuptools pip

COPY requirements.txt /home/jupyter/


RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--port=5555","--allow-root"]

